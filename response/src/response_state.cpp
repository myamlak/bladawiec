// The displacement-sweep state: reference validation, the per-coordinate
// solver set, and the rebase that keeps every stored answer.

#include "qcx/response/response_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

namespace qcx::response {

namespace {

qcx::Error MakeError(std::string message) {
    return qcx::Error{qcx::ErrorCode::kInvalidArgument, std::move(message)};
}

// The density the response differentiates is D = 2 C_occ C_occ^T. The
// reference is one state only when the two returned objects agree on it; a
// wrong occupied count shows up here as well.
constexpr double kDensityConsistencyTolerance = 1e-8;

// The Fock matrix's symmetry is scaled by its largest entry: an asymmetric
// Fock is a defect, not a small number.
constexpr double kFockAsymmetryTolerance = 1e-8;

// Validates everything a response reads, and returns the reference identity
// the screening digest is checked against.
qcx::Result<ResponseReferenceIdentity> ValidateReference(const ResponseStateReference& reference,
                                                         const ResponseScreening& screening) {
    if (!reference.converged)
    {
        return std::unexpected(
            MakeError("response state: the reference must be a converged SCF result; this one "
                      "reports converged = false, so its last iterate is not the stationary "
                      "point a response is solved at"));
    }

    if (reference.numOccupied == 0 || reference.numVirtual == 0)
    {
        return std::unexpected(
            MakeError("response state: the reference needs at least one occupied and one virtual "
                      "orbital; the request states " +
                      std::to_string(reference.numOccupied) + " occupied and " +
                      std::to_string(reference.numVirtual) + " virtual"));
    }

    const std::size_t orbitals = reference.numOccupied + reference.numVirtual;
    const std::size_t numBasisFunctions = reference.numBasisFunctions;

    if (numBasisFunctions < orbitals)
    {
        return std::unexpected(
            MakeError("response state: the basis-function count cannot be below the orbital "
                      "count; the request states " +
                      std::to_string(numBasisFunctions) + " basis functions and " +
                      std::to_string(orbitals) + " orbitals"));
    }

    if (reference.coordinatesBohr.empty() || reference.coordinatesBohr.size() % 3 != 0)
    {
        return std::unexpected(
            MakeError("response state: the nuclear coordinates must number three per atom and "
                      "at least one atom; the request carries " +
                      std::to_string(reference.coordinatesBohr.size()) + " values"));
    }

    if (reference.moCoefficients.size() != numBasisFunctions * orbitals)
    {
        return std::unexpected(
            MakeError("response state: the MO coefficients must be numBasisFunctions x "
                      "numOrbitals = " +
                      std::to_string(numBasisFunctions) + " x " + std::to_string(orbitals) +
                      "; the request carries " + std::to_string(reference.moCoefficients.size()) +
                      " values"));
    }

    if (reference.orbitalEnergies.size() != orbitals)
    {
        return std::unexpected(
            MakeError("response state: the orbital energies must have numOrbitals = " +
                      std::to_string(orbitals) + " entries; the request carries " +
                      std::to_string(reference.orbitalEnergies.size())));
    }

    if (reference.density.size() != numBasisFunctions * numBasisFunctions)
    {
        return std::unexpected(
            MakeError("response state: the density must be numBasisFunctions^2 = " +
                      std::to_string(numBasisFunctions * numBasisFunctions) +
                      " values; the request carries " + std::to_string(reference.density.size())));
    }

    if (reference.fock.size() != numBasisFunctions * numBasisFunctions)
    {
        return std::unexpected(
            MakeError("response state: the Fock matrix must be numBasisFunctions^2 = " +
                      std::to_string(numBasisFunctions * numBasisFunctions) +
                      " values; the request carries " + std::to_string(reference.fock.size())));
    }

    if (reference.moTwoElectron.size() != orbitals * orbitals * orbitals * orbitals)
    {
        return std::unexpected(MakeError(
            "response state: the MO two-electron tensor must be numOrbitals^4 = " +
            std::to_string(orbitals * orbitals * orbitals * orbitals) +
            " values; the request carries " + std::to_string(reference.moTwoElectron.size())));
    }

    double worstDensityResidual = 0.0;

    for (std::size_t mu = 0; mu < numBasisFunctions; ++mu)
    {
        for (std::size_t nu = 0; nu < numBasisFunctions; ++nu)
        {
            double projected = 0.0;

            for (std::size_t orbital = 0; orbital < reference.numOccupied; ++orbital)
            {
                projected += 2.0 * reference.moCoefficients[mu * orbitals + orbital] *
                             reference.moCoefficients[nu * orbitals + orbital];
            }

            worstDensityResidual =
                std::max(worstDensityResidual,
                         std::abs(projected - reference.density[mu * numBasisFunctions + nu]));
        }
    }

    if (worstDensityResidual > kDensityConsistencyTolerance)
    {
        return std::unexpected(
            MakeError("response state: the density is not 2 C_occ C_occ^T for " +
                      std::to_string(reference.numOccupied) +
                      " occupied orbitals (worst element |D - 2 C_occ C_occ^T| = " +
                      std::to_string(worstDensityResidual) +
                      "); the returned density and orbitals describe different states"));
    }

    double largestFockEntry = 0.0;
    double worstFockAsymmetry = 0.0;

    for (std::size_t mu = 0; mu < numBasisFunctions; ++mu)
    {
        for (std::size_t nu = 0; nu < numBasisFunctions; ++nu)
        {
            largestFockEntry =
                std::max(largestFockEntry, std::abs(reference.fock[mu * numBasisFunctions + nu]));
            worstFockAsymmetry = std::max(worstFockAsymmetry,
                                          std::abs(reference.fock[mu * numBasisFunctions + nu] -
                                                   reference.fock[nu * numBasisFunctions + mu]));
        }
    }

    if (worstFockAsymmetry > kFockAsymmetryTolerance * std::max(1.0, largestFockEntry))
    {
        return std::unexpected(MakeError(
            "response state: the Fock matrix is not symmetric (worst |F(mu,nu) - "
            "F(nu,mu)| = " +
            std::to_string(worstFockAsymmetry) + " against a largest entry of " +
            std::to_string(largestFockEntry) + "); the reference is not one Fock matrix"));
    }

    const std::size_t pairCount = numBasisFunctions * (numBasisFunctions + 1) / 2;

    if (screening.pairValues.size() != screening.pairDerivativeBounds.size())
    {
        return std::unexpected(MakeError(
            "response state: the screening structure's two bounds must describe the "
            "same pair set; pairValues has " +
            std::to_string(screening.pairValues.size()) + " entries and pairDerivativeBounds has " +
            std::to_string(screening.pairDerivativeBounds.size())));
    }

    if (!screening.pairValues.empty() && screening.pairValues.size() != pairCount)
    {
        return std::unexpected(MakeError(
            "response state: the screening structure must carry one value per AO pair "
            "= n (n + 1) / 2 = " +
            std::to_string(pairCount) + " entries at " + std::to_string(numBasisFunctions) +
            " basis functions; it carries " + std::to_string(screening.pairValues.size())));
    }

    auto identity = MakeResponseReferenceIdentity(reference.coordinatesBohr, reference.density);

    if (!identity.has_value())
    {
        return std::unexpected(identity.error());
    }

    if (screening.geometry != identity->geometry)
    {
        return std::unexpected(
            MakeError("response state: the screening structure's geometry digest does not match "
                      "the reference coordinates, so its bounds belong to another geometry; "
                      "rebuild or restamp the screening structure for this reference"));
    }

    return identity;
}

} // namespace

ResponseState::ResponseState() = default;

qcx::Result<void> ResponseState::Install(ResponseStateReference reference,
                                         double exchangeFraction,
                                         std::optional<KohnShamKernelTerm> kernel) {
    // The copies are built first and the operator is created against them, so
    // a refusal leaves the state exactly as it was; the vectors are then moved
    // into their members, which transfers the same buffers the operator's
    // spans point at.
    std::vector<double> coordinates(reference.coordinatesBohr.begin(),
                                    reference.coordinatesBohr.end());
    std::vector<double> coefficients(reference.moCoefficients.begin(),
                                     reference.moCoefficients.end());
    std::vector<double> energies(reference.orbitalEnergies.begin(),
                                 reference.orbitalEnergies.end());
    std::vector<double> density(reference.density.begin(), reference.density.end());
    std::vector<double> fock(reference.fock.begin(), reference.fock.end());
    std::vector<double> moTwoElectron(reference.moTwoElectron.begin(),
                                      reference.moTwoElectron.end());

    const ResponseLayout layout{reference.numOccupied, reference.numVirtual};
    // The class's own OrbitalEnergies accessor shadows the type name here.
    const qcx::response::OrbitalEnergies energiesView{std::span<const double>(energies)};
    const MoTwoElectronTensor tensorView{std::span<const double>(moTwoElectron)};

    // The operator the state installs is the method's: the kernel term turns
    // the Hartree-Fock action into the Kohn-Sham one, and its absent case is
    // today's operator, unchanged.
    std::optional<OrbitalHessianOperator> action;
    std::optional<KohnShamOrbitalHessian> kernelAction;

    if (kernel.has_value())
    {
        auto built = KohnShamOrbitalHessian::Create(
            layout, energiesView, tensorView, exchangeFraction, std::move(*kernel));

        if (!built.has_value())
        {
            return std::unexpected(built.error());
        }

        kernelAction.emplace(std::move(*built));
    } else
    {
        auto built =
            OrbitalHessianOperator::Create(layout, energiesView, tensorView, exchangeFraction);

        if (!built.has_value())
        {
            return std::unexpected(built.error());
        }

        action.emplace(std::move(*built));
    }

    _layout = layout;
    _numBasisFunctions = reference.numBasisFunctions;
    _coordinatesBohr = std::move(coordinates);
    _moCoefficients = std::move(coefficients);
    _orbitalEnergies = std::move(energies);
    _density = std::move(density);
    _fock = std::move(fock);
    _moTwoElectron = std::move(moTwoElectron);
    _operator = std::move(action);
    _ksOperator = std::move(kernelAction);

    return {};
}

qcx::Result<ResponseState> ResponseState::Create(ResponseStateReference reference,
                                                 ResponseScreening screening,
                                                 ResponseSolveOptions solverOptions,
                                                 double exchangeFraction,
                                                 std::optional<KohnShamKernelTerm> kernel) {
    auto identity = ValidateReference(reference, screening);

    if (!identity.has_value())
    {
        return std::unexpected(identity.error());
    }

    ResponseState state;
    state._solverOptions = solverOptions;
    state._identity = *identity;
    state._screening = std::move(screening);
    state._history = reference.history;

    auto installed = state.Install(reference, exchangeFraction, std::move(kernel));

    if (!installed.has_value())
    {
        return std::unexpected(installed.error());
    }

    const std::size_t displacements = state.NumDisplacements();
    state._solvers.reserve(displacements);

    for (std::size_t ordinal = 0; ordinal < displacements; ++ordinal)
    {
        state._solvers.emplace_back(solverOptions);
    }

    return state;
}

qcx::Result<void> ResponseState::Rebase(ResponseStateReference reference,
                                        ResponseScreening screening,
                                        double exchangeFraction,
                                        std::optional<KohnShamKernelTerm> kernel) {
    auto identity = ValidateReference(reference, screening);

    if (!identity.has_value())
    {
        return std::unexpected(identity.error());
    }

    if (reference.numBasisFunctions != _numBasisFunctions)
    {
        return std::unexpected(
            MakeError("response state: a rebase cannot change the basis-function count (this "
                      "state carries " +
                      std::to_string(_numBasisFunctions) + ", the new reference states " +
                      std::to_string(reference.numBasisFunctions) +
                      "), because the stored solution vectors are shaped by it"));
    }

    if (reference.numOccupied != _layout.numOccupied || reference.numVirtual != _layout.numVirtual)
    {
        return std::unexpected(
            MakeError("response state: a rebase cannot change the occupied/virtual layout (this "
                      "state carries " +
                      std::to_string(_layout.numOccupied) + " occupied and " +
                      std::to_string(_layout.numVirtual) + " virtual; the new reference states " +
                      std::to_string(reference.numOccupied) + " occupied and " +
                      std::to_string(reference.numVirtual) +
                      " virtual), because every stored solution vector is a response amplitude "
                      "of the old layout"));
    }

    if (reference.coordinatesBohr.size() != _coordinatesBohr.size())
    {
        return std::unexpected(
            MakeError("response state: a rebase cannot change the atom count (this state carries " +
                      std::to_string(NumAtoms()) + " atoms, the new reference states " +
                      std::to_string(reference.coordinatesBohr.size() / 3) +
                      "), because the per-coordinate solver set is keyed on it"));
    }

    // Install copies and rebuilds first, so a refusal leaves the state
    // untouched; only a successful install replaces the reference record.
    auto installed = Install(reference, exchangeFraction, std::move(kernel));

    if (!installed.has_value())
    {
        return std::unexpected(installed.error());
    }

    _identity = *identity;
    _screening = std::move(screening);
    _history = reference.history;
    return {};
}

qcx::Result<ResponseStateSolution> ResponseState::SolveDisplacement(
    NuclearDisplacement displacement,
    const NuclearDisplacementRightHandSide& rightHandSide,
    std::span<const double> moOverlapDerivative) {
    if (displacement.atom >= NumAtoms())
    {
        return std::unexpected(MakeError("response state: displacement atom " +
                                         std::to_string(displacement.atom) + " is outside the " +
                                         std::to_string(NumAtoms()) + " atoms of the reference"));
    }

    if (displacement.direction >= 3)
    {
        return std::unexpected(MakeError("response state: displacement direction " +
                                         std::to_string(displacement.direction) +
                                         " is outside the three Cartesian axes"));
    }

    if (rightHandSide.Layout().numOccupied != _layout.numOccupied ||
        rightHandSide.Layout().numVirtual != _layout.numVirtual)
    {
        return std::unexpected(
            MakeError("response state: the right-hand side's occupied/virtual layout is not this "
                      "state's (" +
                      std::to_string(rightHandSide.Layout().numOccupied) + " occupied and " +
                      std::to_string(rightHandSide.Layout().numVirtual) + " virtual against " +
                      std::to_string(_layout.numOccupied) + " and " +
                      std::to_string(_layout.numVirtual) + ")"));
    }

    if (rightHandSide.Dimension() != _layout.Dimension())
    {
        return std::unexpected(
            MakeError("response state: the right-hand side's dimension is not this state's"));
    }

    const std::size_t ordinal = 3 * displacement.atom + displacement.direction;
    const std::size_t dimension = _layout.Dimension();

    // The solve entry returns the response rather than the report, so the
    // warm-startedness and the iteration count are read from the solver's own
    // accessors. Both are exact: the solve starts from a stored vector exactly
    // when the ordinal's previous answer has this problem's dimension, and the
    // last history entry is this solve's own count.
    const bool warm =
        _solverOptions.reusePrevious && _solvers[ordinal].PreviousSolution(0).size() == dimension;

    FockDerivativeRequest request;
    request.layout = _layout;
    request.displacement = displacement;
    request.numBasisFunctions = _numBasisFunctions;
    request.moCoefficients = _moCoefficients;
    request.density = _density;

    // The Kohn-Sham operator carries the kernel term inside its base action,
    // so the solve reads one operator whichever form the state installed.
    const OrbitalHessianOperator& hessian =
        _ksOperator.has_value() ? _ksOperator->BaseOperator() : *_operator;

    auto solved = SolveNuclearDisplacementResponse(
        hessian, rightHandSide, request, moOverlapDerivative, _identity, _solvers[ordinal]);

    if (!solved.has_value())
    {
        return std::unexpected(solved.error());
    }

    const std::size_t iterations = _solvers[ordinal].ConvergenceHistory().back();
    ++_report.solves;

    if (warm)
    {
        ++_report.warmStarted;
    }

    _report.iterations += iterations;

    ResponseStateSolution solution;
    solution.response = std::move(*solved);
    solution.warmStarted = warm;
    solution.iterations = iterations;
    return solution;
}

} // namespace qcx::response
