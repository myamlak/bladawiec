#include "qcx/response/response_gradient.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace qcx::response {

namespace {

constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

/// Folds one array's bits into a running FNV-1a digest.
///
/// The values are hashed as their raw bits rather than as rounded numbers: a
/// reference that moved by less than a rounding step is still a different
/// reference, and a digest is only useful here if it is finer than the thing it
/// guards.
/// \param digest The running digest.
/// \param values The array to fold in.
/// \returns The updated digest.
std::uint64_t FoldDigest(std::uint64_t digest, std::span<const double> values) {
    for (const double value : values)
    {
        std::uint64_t bits = std::bit_cast<std::uint64_t>(value);

        for (int byte = 0; byte < 8; ++byte)
        {
            digest ^= static_cast<std::uint64_t>(bits & 0xFFU);
            digest *= kFnvPrime;
            bits >>= 8;
        }
    }

    return digest;
}

/// Validates the occupied/virtual structure, as the operator's own factory does.
/// \param layout The layout to check.
/// \returns An Error (kInvalidArgument) when a block is empty.
qcx::Result<void> ValidateLayout(const ResponseLayout& layout) {
    if (layout.numOccupied == 0 || layout.numVirtual == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "a response right-hand side needs at least one occupied and one virtual "
                       "orbital"});
    }

    return {};
}

} // namespace

std::string ToString(const NuclearDisplacement& displacement) {
    static const char* const kAxisNames[3] = {"x", "y", "z"};
    const std::string axis = displacement.direction < 3 ? kAxisNames[displacement.direction] : "?";

    return "atom" + std::to_string(displacement.atom) + "-" + axis;
}

qcx::Result<ResponseReferenceIdentity> MakeResponseReferenceIdentity(
    std::span<const double> coordinatesBohr, std::span<const double> density) {
    if (coordinatesBohr.empty())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "a reference identity needs the nuclear coordinates"});
    }

    if (density.empty())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "a reference identity needs the reference density"});
    }

    if (coordinatesBohr.size() % 3 != 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the nuclear coordinates must number three per atom"});
    }

    ResponseReferenceIdentity identity;
    identity.geometry = FoldDigest(kFnvOffsetBasis, coordinatesBohr);
    identity.density = FoldDigest(kFnvOffsetBasis, density);

    return identity;
}

NuclearDisplacementRightHandSide::NuclearDisplacementRightHandSide(ResponseLayout layout,
                                                                   std::size_t numBasisFunctions) :
    _layout(layout), _numBasisFunctions(numBasisFunctions) {}

qcx::Result<NuclearDisplacementRightHandSide> NuclearDisplacementRightHandSide::Create(
    ResponseLayout layout, std::size_t numBasisFunctions) {
    auto layoutStatus = ValidateLayout(layout);

    if (!layoutStatus.has_value())
    {
        return std::unexpected(layoutStatus.error());
    }

    const std::size_t numOrbitals = layout.numOccupied + layout.numVirtual;

    if (numBasisFunctions < numOrbitals)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the basis-function count cannot be below the orbital count"});
    }

    return NuclearDisplacementRightHandSide(layout, numBasisFunctions);
}

qcx::Result<void> NuclearDisplacementRightHandSide::Add(std::string_view name,
                                                        FockDerivativeContributionFn contribution) {
    if (name.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "a contribution needs a name"});
    }

    if (contribution == nullptr)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the contribution named '" + std::string(name) + "' is an empty callback"});
    }

    for (const std::string& existing : _names)
    {
        if (existing == name)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "the contribution name '" + std::string(name) + "' is already taken"});
        }
    }

    _names.emplace_back(name);
    _contributions.push_back(std::move(contribution));

    return {};
}

qcx::Result<void> NuclearDisplacementRightHandSide::Assemble(
    const FockDerivativeRequest& request, std::span<double> occupiedVirtualBlock) const {
    if (occupiedVirtualBlock.size() != Dimension())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the occupied-virtual block must be sized numOccupied * "
                                          "numVirtual"});
    }

    if (request.layout.numOccupied != _layout.numOccupied ||
        request.layout.numVirtual != _layout.numVirtual ||
        request.numBasisFunctions != _numBasisFunctions)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "the request does not describe the layout this right-hand side was built for"});
    }

    std::fill(occupiedVirtualBlock.begin(), occupiedVirtualBlock.end(), 0.0);

    for (const FockDerivativeContributionFn& contribution : _contributions)
    {
        auto status = contribution(request, occupiedVirtualBlock);

        if (!status.has_value())
        {
            return std::unexpected(status.error());
        }
    }

    return {};
}

qcx::Result<void> OrbitalResponseCache::Store(OrbitalResponse response) {
    if (response.numOrbitals == 0 || response.amplitudes.empty() ||
        response.perturbedDensity.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "an empty response cannot be cached"});
    }

    if (response.reference.geometry == 0 && response.reference.density == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "a cached response must carry the reference it was solved at"});
    }

    const ResponseCacheHandle handle{response.displacement, response.reference};

    for (OrbitalResponse& entry : _entries)
    {
        if (entry.displacement.atom == handle.displacement.atom &&
            entry.displacement.direction == handle.displacement.direction)
        {
            entry = std::move(response);
            return {};
        }
    }

    _entries.push_back(std::move(response));

    return {};
}

const OrbitalResponse* OrbitalResponseCache::Lookup(const ResponseCacheHandle& handle) const {
    for (const OrbitalResponse& entry : _entries)
    {
        if (entry.displacement.atom != handle.displacement.atom ||
            entry.displacement.direction != handle.displacement.direction)
        {
            continue;
        }

        // The reference is what makes this a hit rather than a stale read. A
        // response solved at another geometry, or against another density, is
        // not an approximation of this one - it is the answer to a different
        // question, and returning it would leave the energy right and the
        // gradient wrong.
        if (!(entry.reference == handle.reference))
        {
            return nullptr;
        }

        return &entry;
    }

    return nullptr;
}

qcx::Result<double> EnergyWeightedTerm(std::span<const double> fock,
                                       const OrbitalResponse& response) {
    const std::size_t numBasisFunctions = response.numBasisFunctions;

    if (fock.size() != numBasisFunctions * numBasisFunctions)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the Fock matrix must be numBasisFunctions squared"});
    }

    if (response.perturbedDensity.size() != numBasisFunctions * numBasisFunctions)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the perturbed density must be numBasisFunctions squared"});
    }

    // Tr(F D(1)), with the Fock matrix symmetric: the density is indexed by its
    // column so that both matrices are read in their own row-major order.
    double term = 0.0;

    for (std::size_t mu = 0; mu < numBasisFunctions; ++mu)
    {
        for (std::size_t nu = 0; nu < numBasisFunctions; ++nu)
        {
            term += fock[mu * numBasisFunctions + nu] *
                    response.perturbedDensity[nu * numBasisFunctions + mu];
        }
    }

    return term;
}

qcx::Result<void> OrbitalGradient(std::span<const double> moFock,
                                  ResponseLayout layout,
                                  std::span<double> orbitalGradient) {
    auto layoutStatus = ValidateLayout(layout);

    if (!layoutStatus.has_value())
    {
        return std::unexpected(layoutStatus.error());
    }

    const std::size_t numOrbitals = layout.numOccupied + layout.numVirtual;

    if (moFock.size() != numOrbitals * numOrbitals)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the MO Fock matrix must be the orbital count squared"});
    }

    if (orbitalGradient.size() != layout.Dimension())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the orbital gradient must be sized numOccupied * "
                                          "numVirtual"});
    }

    for (std::size_t i = 0; i < layout.numOccupied; ++i)
    {
        for (std::size_t a = 0; a < layout.numVirtual; ++a)
        {
            orbitalGradient[i * layout.numVirtual + a] =
                2.0 * moFock[(layout.numOccupied + a) * numOrbitals + i];
        }
    }

    return {};
}

qcx::Result<OrbitalResponse> SolveNuclearDisplacementResponse(
    const OrbitalHessianOperator& hessian,
    const NuclearDisplacementRightHandSide& rightHandSide,
    const FockDerivativeRequest& request,
    std::span<const double> moOverlapDerivative,
    const ResponseReferenceIdentity& reference,
    ResponseSolver<OrbitalHessianOperator>& solver) {
    const ResponseLayout& layout = rightHandSide.Layout();
    const std::size_t dimension = layout.Dimension();
    const std::size_t numOccupied = layout.numOccupied;
    const std::size_t numVirtual = layout.numVirtual;
    const std::size_t numOrbitals = numOccupied + numVirtual;
    const std::size_t numBasisFunctions = request.numBasisFunctions;

    if (hessian.Layout().numOccupied != numOccupied || hessian.Layout().numVirtual != numVirtual)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the operator's layout is not the right-hand side's"});
    }

    if (moOverlapDerivative.size() != numOrbitals * numOrbitals)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the MO overlap derivative must be the orbital count squared"});
    }

    if (request.moCoefficients.size() != numBasisFunctions * numOrbitals)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the MO coefficients must be numBasisFunctions x "
                                          "numOrbitals"});
    }

    if (request.density.size() != numBasisFunctions * numBasisFunctions)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the reference density must be numBasisFunctions "
                                          "squared"});
    }

    // Step 1: B, the occupied-virtual block of the derivative Fock matrix.
    std::vector<double> derivativeFock(dimension, 0.0);

    // Step 2: A U = -B, with the sign flipped in this one place. The response
    // equation's right-hand side is the negative of the derivative Fock block;
    // the assembled object above is named for the physical quantity, so the
    // flip happens here rather than being folded into every contribution.
    auto assembleStatus = rightHandSide.Assemble(request, derivativeFock);

    if (!assembleStatus.has_value())
    {
        return std::unexpected(assembleStatus.error());
    }

    std::vector<double> rhs(dimension, 0.0);

    for (std::size_t k = 0; k < dimension; ++k)
    {
        rhs[k] = -derivativeFock[k];
    }

    std::vector<double> amplitudes(dimension, 0.0);
    const auto solved = solver.Solve(hessian, 1, rhs, amplitudes);

    if (!solved.has_value())
    {
        return std::unexpected(solved.error());
    }

    // Step 3: the full rotation. Orthonormality is a constraint on the whole
    // matrix, not a choice: the perturbed orbitals are C (1 + U), so
    // (C (1 + U))^T S(R) (C (1 + U)) = 1 to first order requires
    // U + U^T = -C^T S^R C. The solve supplies the antisymmetric half of the
    // occupied-virtual block; the redundant rotations and the symmetric half
    // are what that condition leaves.
    //
    // This is the placement decision the decomposition forces, stated where it
    // is made: the overlap derivative enters the response through U here, so a
    // caller must NOT also carry it in the fixed-reference part of its gradient.
    std::vector<double> full(numOrbitals * numOrbitals, 0.0);

    const auto overlapEntry = [&](std::size_t row, std::size_t column) {
        return moOverlapDerivative[row * numOrbitals + column];
    };

    for (std::size_t i = 0; i < numOccupied; ++i)
    {
        for (std::size_t j = 0; j < numOccupied; ++j)
        {
            full[i * numOrbitals + j] = -0.5 * overlapEntry(i, j);
        }
    }

    for (std::size_t a = 0; a < numVirtual; ++a)
    {
        for (std::size_t b = 0; b < numVirtual; ++b)
        {
            const std::size_t row = numOccupied + a;
            const std::size_t column = numOccupied + b;
            full[row * numOrbitals + column] = -0.5 * overlapEntry(row, column);
        }
    }

    for (std::size_t i = 0; i < numOccupied; ++i)
    {
        for (std::size_t a = 0; a < numVirtual; ++a)
        {
            const std::size_t virtualOrbital = numOccupied + a;
            const double amplitude = amplitudes[i * numVirtual + a];
            const double symmetric = -0.5 * overlapEntry(virtualOrbital, i);

            full[virtualOrbital * numOrbitals + i] = amplitude + symmetric;
            full[i * numOrbitals + virtualOrbital] = -amplitude + symmetric;
        }
    }

    // Step 4: the perturbed density, from the whole of U. With P the occupied
    // projector, D(1) = 2 (U P + P U^T), which for a diagonal P reads
    // D(1)_pq = 2 (U_pq [q occupied] + U_qp [p occupied]).
    std::vector<double> moDensity(numOrbitals * numOrbitals, 0.0);

    for (std::size_t p = 0; p < numOrbitals; ++p)
    {
        for (std::size_t q = 0; q < numOrbitals; ++q)
        {
            const double fromRow = q < numOccupied ? full[p * numOrbitals + q] : 0.0;
            const double fromColumn = p < numOccupied ? full[q * numOrbitals + p] : 0.0;
            moDensity[p * numOrbitals + q] = 2.0 * (fromRow + fromColumn);
        }
    }

    OrbitalResponse response;
    response.numBasisFunctions = numBasisFunctions;
    response.numOrbitals = numOrbitals;
    response.displacement = request.displacement;
    response.reference = reference;
    response.amplitudes = std::move(full);
    response.perturbedDensity.assign(numBasisFunctions * numBasisFunctions, 0.0);

    // D(1)_AO = C D(1)_MO C^T.
    for (std::size_t mu = 0; mu < numBasisFunctions; ++mu)
    {
        for (std::size_t nu = 0; nu < numBasisFunctions; ++nu)
        {
            double value = 0.0;

            for (std::size_t p = 0; p < numOrbitals; ++p)
            {
                const double left = request.moCoefficients[mu * numOrbitals + p];

                if (left == 0.0)
                {
                    continue;
                }

                for (std::size_t q = 0; q < numOrbitals; ++q)
                {
                    value += left * moDensity[p * numOrbitals + q] *
                             request.moCoefficients[nu * numOrbitals + q];
                }
            }

            response.perturbedDensity[mu * numBasisFunctions + nu] = value;
        }
    }

    return response;
}

} // namespace qcx::response
