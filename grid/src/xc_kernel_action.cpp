#include "qcx/grid/xc_kernel_action.hpp"

#include "internal/xc_point_assembly.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qcx::grid {

namespace {

using internal::ContractSelectedSpinDensity;
using internal::PointDensity;

// The second-derivative tier answers for the components the functional reads,
// so the active set is the required mask in identifier order.
std::vector<excgrid::Component> ActiveComponents(const excgrid::ComponentMask& required) {
    std::vector<excgrid::Component> active;

    for (std::uint8_t bit = 0; bit < 8; ++bit)
    {
        const auto id = static_cast<excgrid::Component>(bit);

        if (required.Test(id))
        {
            active.push_back(id);
        }
    }

    return active;
}

// The component names a refusal lists, in identifier order.
std::string ComponentNames(const excgrid::ComponentMask& mask) {
    std::string names;

    for (std::uint8_t bit = 0; bit < 8; ++bit)
    {
        const auto id = static_cast<excgrid::Component>(bit);

        if (!mask.Test(id))
        {
            continue;
        }

        if (!names.empty())
        {
            names += ", ";
        }

        names += std::string(excgrid::Describe(id).name);
    }

    return names;
}

// The refusal a functional without a usable tier earns, or an empty string.
std::string TierRefusal(const excgrid::XcFunctional& functional) {
    const excgrid::ComponentMask required = functional.RequiredMask();
    const excgrid::ComponentMask spanned = functional.SecondDerivativeMask();

    if (required.bits == 0)
    {
        return "the functional reads no components";
    }

    const excgrid::KernelStatus status =
        excgrid::SecondDerivativeStatus(required, required, spanned);

    if (status == excgrid::KernelStatus::kRefusedUnsupportedCapability)
    {
        return "the functional has no second-derivative tier";
    }

    if (status == excgrid::KernelStatus::kRefusedSecondDerivativeCoverage)
    {
        const excgrid::ComponentMask missing =
            excgrid::UnspannedRequestComponents(required, required, spanned);

        return "the second-derivative tier does not span " + ComponentNames(missing);
    }

    // A functional naming a component outside the collinear supply cannot be
    // handed its inputs, whatever its tier covers.
    const excgrid::ComponentMask outside =
        excgrid::ComponentMask{required.bits & ~excgrid::ComponentMask::Collinear().bits};

    if (outside.bits != 0)
    {
        return "the functional reads a component outside the collinear supply: " +
               ComponentNames(outside);
    }

    return {};
}

// One point's density value and gradient invariants, from the reference field.
double ComponentValue(excgrid::Component id, const PointDensity& field, double sigma) {
    switch (id)
    {
    case excgrid::Component::RhoA:
    case excgrid::Component::RhoB:
        return field.rho;
    case excgrid::Component::SigmaAa:
    case excgrid::Component::SigmaAb:
    case excgrid::Component::SigmaBb:
        return sigma;
    case excgrid::Component::TauA:
    case excgrid::Component::TauB:
        return field.tau;
    default:
        return 0.0;
    }
}

// The trial vector's total response at one point, as the three pair sums the
// kernel contracts: the density value, the response gradient against the
// density gradient, and the kinetic-energy density.
struct DensityResponse {
    double u = 0.0;
    double gradientDirection = 0.0;
    double tau = 0.0;
};

// One component of the density response to a unit amplitude x: the component's
// derivative in the same identifier order the kernel expects.
double ResponseComponent(excgrid::Component id, const DensityResponse& response) {
    switch (id)
    {
    case excgrid::Component::RhoA:
    case excgrid::Component::RhoB:
        return 2.0 * response.u;
    case excgrid::Component::SigmaAa:
    case excgrid::Component::SigmaAb:
    case excgrid::Component::SigmaBb:
        return 4.0 * response.gradientDirection;
    case excgrid::Component::TauA:
    case excgrid::Component::TauB:
        return response.tau;
    default:
        return 0.0;
    }
}

} // namespace

// The parameter order is the constructor's declaration in the class, where
// each call site reads it by name; BEGIN/END because the diagnostic lands on a
// later parameter line of the multi-line definition.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
XcKernelAction::XcKernelAction(std::span<const excgrid::Block> blocks,
                               const AoEvaluator& evaluator,
                               const excgrid::XcFunctional& functional,
                               std::span<const ShellEnvelope> envelopes,
                               Eigen::MatrixXd moCoefficients,
                               Eigen::MatrixXd spinDensity,
                               std::vector<double> shellWeights,
                               std::vector<excgrid::Component> active,
                               std::size_t numOccupied,
                               std::size_t numVirtual,
                               std::size_t aoCount,
                               double screeningTolerance,
                               bool needsGradient) noexcept :
    _blocks(blocks), _evaluator(&evaluator), _functional(&functional), _envelopes(envelopes),
    _moCoefficients(std::move(moCoefficients)), _spinDensity(std::move(spinDensity)),
    _shellWeights(std::move(shellWeights)), _active(std::move(active)), _numOccupied(numOccupied),
    _numVirtual(numVirtual), _numOrbitals(numOccupied + numVirtual), _aoCount(aoCount),
    _screeningTolerance(screeningTolerance), _needsGradient(needsGradient) {}

// NOLINTEND(bugprone-easily-swappable-parameters)

qcx::Result<XcKernelAction> XcKernelAction::Create(const XcGridEngine& engine,
                                                   const Eigen::MatrixXd& moCoefficients,
                                                   const Eigen::MatrixXd& density,
                                                   std::size_t numOccupied,
                                                   std::size_t numVirtual,
                                                   const XcKernelSettings& settings) {
    return Create(engine.Grid().Blocks(),
                  engine.Evaluator(),
                  engine.Functional(),
                  engine.ShellEnvelopes(),
                  moCoefficients,
                  density,
                  numOccupied,
                  numVirtual,
                  settings);
}

qcx::Result<XcKernelAction> XcKernelAction::Create(std::span<const excgrid::Block> blocks,
                                                   const AoEvaluator& evaluator,
                                                   const excgrid::XcFunctional& functional,
                                                   std::span<const ShellEnvelope> envelopes,
                                                   const Eigen::MatrixXd& moCoefficients,
                                                   const Eigen::MatrixXd& density,
                                                   std::size_t numOccupied,
                                                   std::size_t numVirtual,
                                                   const XcKernelSettings& settings) {
    const std::size_t aoCount = evaluator.AOCount();
    const std::size_t numOrbitals = numOccupied + numVirtual;

    if (numOccupied == 0 || numVirtual == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "XcKernelAction: numOccupied and numVirtual must both be nonzero"});
    }

    if (numOrbitals > aoCount)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "XcKernelAction: numOccupied + numVirtual exceeds AOCount()"});
    }

    if (moCoefficients.rows() != static_cast<Eigen::Index>(aoCount) ||
        moCoefficients.cols() != static_cast<Eigen::Index>(numOrbitals))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "XcKernelAction: moCoefficients must be AOCount() x (o + v)"});
    }

    if (density.rows() != static_cast<Eigen::Index>(aoCount) ||
        density.cols() != static_cast<Eigen::Index>(aoCount))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "XcKernelAction: density must be AOCount() x AOCount()"});
    }

    if (envelopes.size() != evaluator.ShellRanges().size())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "XcKernelAction: one envelope per evaluator shell is required"});
    }

    if (const std::string refusal = TierRefusal(functional); !refusal.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented, "XcKernelAction: " + refusal});
    }

    const excgrid::ComponentMask required = functional.RequiredMask();
    std::vector<excgrid::Component> active = ActiveComponents(required);

    const Eigen::MatrixXd spinDensity = 0.5 * density;

    std::vector<double> shellWeights;
    const auto weights =
        ShellDensityWeights(envelopes, internal::FlattenRowMajor(spinDensity), aoCount);

    if (!weights.has_value())
    {
        return std::unexpected(weights.error());
    }

    shellWeights = *weights;

    const bool needsGradient = functional.UsesGradient() ||
                               required.Test(excgrid::Component::TauA) ||
                               required.Test(excgrid::Component::TauB);

    return XcKernelAction(blocks,
                          evaluator,
                          functional,
                          envelopes,
                          moCoefficients,
                          spinDensity,
                          std::move(shellWeights),
                          std::move(active),
                          numOccupied,
                          numVirtual,
                          aoCount,
                          settings.screeningTolerance,
                          needsGradient);
}

qcx::Result<void> XcKernelAction::Apply(std::span<const double> amplitudes,
                                        std::span<double> result) const {
    if (amplitudes.size() != Dimension() || result.size() != Dimension())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "XcKernelAction::Apply: both vectors must hold o * v entries"});
    }

    std::fill(result.begin(), result.end(), 0.0);

    if (Dimension() == 0)
    {
        return {};
    }

    const std::size_t nAO = _aoCount;
    const std::size_t nOrb = _numOrbitals;

    // The unselected slots keep whatever they hold: zeroing the gradient
    // scratch once is what makes an LDA's unused gradient and tau read as zero
    // rather than as a stale value from the point before.
    std::vector<double> values(nAO, 0.0);
    std::vector<double> gradients(3 * nAO, 0.0);
    std::vector<double> valueContraction(nAO, 0.0);
    std::vector<double> gradientContraction(3 * nAO, 0.0);
    std::vector<double> moValue(nOrb, 0.0);
    std::vector<double> moGradient(3 * nOrb, 0.0);

    std::vector<std::size_t> selectedShells;
    std::vector<std::size_t> selectedAos;

    const std::size_t active = _active.size();
    std::vector<double> rightHandSide(active, 0.0);

    excgrid::PointInputs inputs;
    excgrid::PointResult point;
    excgrid::PointSecondDerivative second;

    std::array<double, 3> gradientTotal{};

    for (const excgrid::Block& block : _blocks)
    {
        for (std::size_t p = 0; p < block.pointCount; ++p)
        {
            const double weight = block.weights[p];

            if (weight == 0.0)
            {
                continue;
            }

            selectedShells.clear();
            selectedAos.clear();

            for (std::size_t shell = 0; shell < _envelopes.size(); ++shell)
            {
                const ShellEnvelope& envelope = _envelopes[shell];

                if (!ShellIsSignificant(
                        envelope, block.points[p], _shellWeights[shell], _screeningTolerance))
                {
                    continue;
                }

                selectedShells.push_back(shell);

                for (std::size_t slot = 0; slot < envelope.functionCount; ++slot)
                {
                    selectedAos.push_back(envelope.aoOffset + slot);
                }
            }

            if (selectedAos.empty())
            {
                continue;
            }

            if (_needsGradient)
            {
                _evaluator->EvaluateGradientsSelected(
                    block.points[p], selectedShells, values, gradients);
            } else
            {
                _evaluator->EvaluateSelected(block.points[p], selectedShells, values);
            }

            const PointDensity field = ContractSelectedSpinDensity(_spinDensity,
                                                                   values,
                                                                   gradients,
                                                                   selectedAos,
                                                                   valueContraction,
                                                                   gradientContraction);

            // The orbital values and first derivatives over the kept AOs. Only
            // the selected slots of the fetch are meaningful, so the sums run
            // over the selection rather than the whole AO range.
            for (std::size_t orbital = 0; orbital < nOrb; ++orbital)
            {
                moValue[orbital] = 0.0;
                moGradient[3 * orbital + 0] = 0.0;
                moGradient[3 * orbital + 1] = 0.0;
                moGradient[3 * orbital + 2] = 0.0;
            }

            for (const std::size_t mu : selectedAos)
            {
                for (std::size_t orbital = 0; orbital < nOrb; ++orbital)
                {
                    const double coefficient = _moCoefficients(static_cast<Eigen::Index>(mu),
                                                               static_cast<Eigen::Index>(orbital));

                    moValue[orbital] += coefficient * values[mu];

                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        moGradient[3 * orbital + axis] += coefficient * gradients[3 * mu + axis];
                    }
                }
            }

            // The trial vector's total response at this point: the pair sums
            // the kernel contracts over, and the density-gradient response the
            // sigma curvature needs.
            double uTotal = 0.0;
            double tauTotal = 0.0;
            gradientTotal = {0.0, 0.0, 0.0};

            for (std::size_t occupied = 0; occupied < _numOccupied; ++occupied)
            {
                for (std::size_t virtualOrbital = 0; virtualOrbital < _numVirtual; ++virtualOrbital)
                {
                    const double coefficient = amplitudes[occupied * _numVirtual + virtualOrbital];

                    if (coefficient == 0.0)
                    {
                        continue;
                    }

                    const std::size_t i = occupied;
                    const std::size_t a = _numOccupied + virtualOrbital;

                    uTotal += coefficient * moValue[i] * moValue[a];
                    tauTotal += coefficient * (moGradient[3 * a + 0] * moGradient[3 * i + 0] +
                                               moGradient[3 * a + 1] * moGradient[3 * i + 1] +
                                               moGradient[3 * a + 2] * moGradient[3 * i + 2]);

                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        gradientTotal[axis] +=
                            coefficient * (moValue[a] * moGradient[3 * i + axis] +
                                           moGradient[3 * a + axis] * moValue[i]);
                    }
                }
            }

            const double sigma = field.gradient[0] * field.gradient[0] +
                                 field.gradient[1] * field.gradient[1] +
                                 field.gradient[2] * field.gradient[2];
            const double gradientDirection = field.gradient[0] * gradientTotal[0] +
                                             field.gradient[1] * gradientTotal[1] +
                                             field.gradient[2] * gradientTotal[2];

            inputs = excgrid::PointInputs{};

            for (const excgrid::Component id : _active)
            {
                inputs.Set(id, ComponentValue(id, field, sigma));
            }

            for (std::size_t slot = 0; slot < active; ++slot)
            {
                rightHandSide[slot] = ResponseComponent(
                    _active[slot], DensityResponse{uTotal, gradientDirection, tauTotal});
            }

            const excgrid::KernelStatus status = _functional->EvaluatePointWithSecondDerivatives(
                inputs, rightHandSide, point, second);

            if (status != excgrid::KernelStatus::kOk)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInternalError,
                               "XcKernelAction: the kernel refused the reference contraction: " +
                                   std::string(excgrid::DescribeStatus(status))});
            }

            double hRho = 0.0;
            double hSigma = 0.0;
            double hTau = 0.0;
            double sigmaCurvature = 0.0;

            for (std::size_t slot = 0; slot < active; ++slot)
            {
                switch (_active[slot])
                {
                case excgrid::Component::RhoA:
                case excgrid::Component::RhoB:
                    hRho += second.contracted[slot];
                    break;
                case excgrid::Component::SigmaAa:
                case excgrid::Component::SigmaAb:
                case excgrid::Component::SigmaBb:
                    hSigma += second.contracted[slot];
                    sigmaCurvature += point.first[excgrid::IndexOf(_active[slot])];
                    break;
                case excgrid::Component::TauA:
                case excgrid::Component::TauB:
                    hTau += second.contracted[slot];
                    break;
                default:
                    break;
                }
            }

            const double kernelScale = 0.25 * weight;
            const double curvatureScale = 2.0 * weight * sigmaCurvature;

            for (std::size_t occupied = 0; occupied < _numOccupied; ++occupied)
            {
                for (std::size_t virtualOrbital = 0; virtualOrbital < _numVirtual; ++virtualOrbital)
                {
                    const std::size_t i = occupied;
                    const std::size_t a = _numOccupied + virtualOrbital;

                    std::array<double, 3> uGradient{};

                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        uGradient[axis] = moValue[a] * moGradient[3 * i + axis] +
                                          moGradient[3 * a + axis] * moValue[i];
                    }

                    const double u = moValue[i] * moValue[a];
                    const double tau = moGradient[3 * a + 0] * moGradient[3 * i + 0] +
                                       moGradient[3 * a + 1] * moGradient[3 * i + 1] +
                                       moGradient[3 * a + 2] * moGradient[3 * i + 2];
                    const double uGradientDotField = uGradient[0] * field.gradient[0] +
                                                     uGradient[1] * field.gradient[1] +
                                                     uGradient[2] * field.gradient[2];

                    const double kernelPart =
                        2.0 * u * hRho + 4.0 * uGradientDotField * hSigma + tau * hTau;
                    const double curvaturePart = uGradient[0] * gradientTotal[0] +
                                                 uGradient[1] * gradientTotal[1] +
                                                 uGradient[2] * gradientTotal[2];

                    result[occupied * _numVirtual + virtualOrbital] +=
                        kernelScale * kernelPart + curvatureScale * curvaturePart;
                }
            }
        }
    }

    return {};
}

} // namespace qcx::grid
