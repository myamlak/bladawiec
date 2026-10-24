#include "qcx/grid/xc_kernel_gradient.hpp"

#include "internal/xc_point_assembly.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/grid/shell_screening.hpp"
#include "qcx/grid/xc_grid_engine.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace qcx::grid {

namespace {

using internal::ContractSelectedSpinDensity;
using internal::PointDensity;

// The evaluation's hessian order: xx, xy, xz, yy, yz, zz, mu-major.
constexpr std::array<std::array<std::size_t, 2>, 6> kHessianComponent = {
    {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}}};

// The slot value every AO is filled with before the envelopes claim it, so that
// an envelope set that does not cover the basis is refused rather than walked.
constexpr std::size_t kNoAtom = std::numeric_limits<std::size_t>::max();

// Both parts at once: DerivativePart names one bit each. The combination is not
// one of its enumerators, which is what a bitmask is for.
excgrid::DerivativePart BothParts() noexcept {
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    return static_cast<excgrid::DerivativePart>(
        static_cast<std::uint8_t>(excgrid::DerivativePart::kCoordinates) |
        static_cast<std::uint8_t>(excgrid::DerivativePart::kWeights));
}

// A provider refusal's name, for the error the walk returns instead of a block:
// a refused block answered with zeros is a wrong right-hand side that looks
// plausible.
std::string_view DerivativeStatusName(excgrid::DerivativeStatus status) noexcept {
    switch (status)
    {
    case excgrid::DerivativeStatus::kOk:
        return "ok";
    case excgrid::DerivativeStatus::kRefusedSecondOrderUnavailable:
        return "second order unavailable at this build";
    case excgrid::DerivativeStatus::kRefusedBufferTooSmall:
        return "the caller's buffers are too small for the block";
    case excgrid::DerivativeStatus::kRefusedUndifferentiableGrid:
        return "the block is not a point set this geometry's grid produces";
    }

    return "unrecognised status";
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

    const excgrid::ComponentMask outside =
        excgrid::ComponentMask{required.bits & ~excgrid::ComponentMask::Collinear().bits};

    if (outside.bits != 0)
    {
        return "the functional reads a component outside the collinear supply: " +
               ComponentNames(outside);
    }

    return {};
}

// One component's reference value, from the point's density field.
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

// One component's response to the displacement, in the identifier order the
// kernel expects.
double ResponseComponent(excgrid::Component id, double rho, double gradientSigma, double tau) {
    switch (id)
    {
    case excgrid::Component::RhoA:
    case excgrid::Component::RhoB:
        return rho;
    case excgrid::Component::SigmaAa:
    case excgrid::Component::SigmaAb:
    case excgrid::Component::SigmaBb:
        return gradientSigma;
    case excgrid::Component::TauA:
    case excgrid::Component::TauB:
        return tau;
    default:
        return 0.0;
    }
}

// The atom each AO sits on, from the envelopes.
qcx::Result<std::vector<std::size_t>> BuildAtomOfAo(std::span<const ShellEnvelope> envelopes,
                                                    std::size_t aoCount) {
    std::vector<std::size_t> atomOfAo(aoCount, kNoAtom);

    for (const ShellEnvelope& shell : envelopes)
    {
        if (shell.aoOffset + shell.functionCount > aoCount)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "EvaluateXcKernelDerivativeBlock: a shell envelope reaches past "
                           "AOCount()"});
        }

        for (std::size_t ao = 0; ao < shell.functionCount; ++ao)
        {
            std::size_t& slot = atomOfAo[shell.aoOffset + ao];

            if (slot != kNoAtom)
            {
                return std::unexpected(qcx::Error{
                    qcx::ErrorCode::kInvalidArgument,
                    "EvaluateXcKernelDerivativeBlock: two shell envelopes claim the same AO "
                    "slot"});
            }

            slot = shell.atomIndex;
        }
    }

    for (const std::size_t slot : atomOfAo)
    {
        if (slot == kNoAtom)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "EvaluateXcKernelDerivativeBlock: the shell envelopes do not cover "
                           "every AO"});
        }
    }

    return atomOfAo;
}

// One AO's second derivative expanded into a row-major 3x3, from the
// evaluator's six components.
std::array<double, 9> ExpandHessian(std::span<const double> hessians, std::size_t ao) {
    std::array<double, 9> expanded{};

    for (std::size_t component = 0; component < kHessianComponent.size(); ++component)
    {
        const std::size_t row = kHessianComponent[component][0];
        const std::size_t column = kHessianComponent[component][1];
        const double entry = hessians[6 * ao + component];
        // Assignment, not accumulation: the three diagonal components name the
        // same slot twice.
        expanded[3 * row + column] = entry;
        expanded[3 * column + row] = entry;
    }

    return expanded;
}

// The second-derivative tier answers for the components the functional reads.
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

} // namespace

qcx::Result<XcKernelDerivativeBlock> EvaluateXcKernelDerivativeBlock(
    const XcGridEngine& engine,
    const excgrid::GridDerivativeProvider& provider,
    const Eigen::MatrixXd& moCoefficients,
    const Eigen::MatrixXd& density,
    std::size_t numOccupied,
    std::size_t numVirtual,
    const XcKernelSettings& settings) {
    const std::size_t aoCount = engine.AOCount();
    const std::size_t numOrbitals = numOccupied + numVirtual;

    if (numOccupied == 0 || numVirtual == 0)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "EvaluateXcKernelDerivativeBlock: numOccupied and numVirtual must both be nonzero"});
    }

    if (numOrbitals > aoCount)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "EvaluateXcKernelDerivativeBlock: numOccupied + numVirtual exceeds "
                       "AOCount()"});
    }

    if (moCoefficients.rows() != static_cast<Eigen::Index>(aoCount) ||
        moCoefficients.cols() != static_cast<Eigen::Index>(numOrbitals))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "EvaluateXcKernelDerivativeBlock: moCoefficients must be AOCount() x "
                       "(o + v)"});
    }

    if (density.rows() != static_cast<Eigen::Index>(aoCount) ||
        density.cols() != static_cast<Eigen::Index>(aoCount))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "EvaluateXcKernelDerivativeBlock: density must be AOCount() x "
                       "AOCount()"});
    }

    const std::span<const ShellEnvelope> envelopes = engine.ShellEnvelopes();

    if (envelopes.size() != engine.Evaluator().ShellRanges().size())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "EvaluateXcKernelDerivativeBlock: one envelope per evaluator shell is "
                       "required"});
    }

    if (provider.MaxOrder() < excgrid::DerivativeOrder::kFirst)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented,
                       "EvaluateXcKernelDerivativeBlock: the provider supplies no first-order "
                       "derivatives"});
    }

    const excgrid::XcFunctional& functional = engine.Functional();

    if (const std::string refusal = TierRefusal(functional); !refusal.empty())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                          "EvaluateXcKernelDerivativeBlock: " + refusal});
    }

    const std::vector<excgrid::Component> active = ActiveComponents(functional.RequiredMask());
    const std::size_t activeCount = active.size();

    const auto builtAtomOfAo = BuildAtomOfAo(envelopes, aoCount);

    if (!builtAtomOfAo.has_value())
    {
        return std::unexpected(builtAtomOfAo.error());
    }

    const std::vector<std::size_t>& atomOfAo = *builtAtomOfAo;

    // The atom count the request addresses: every atom the grid carries a point
    // for. A molecular grid has a subgrid per atom, so the maximum owning atom
    // index plus one is the molecule's atom count.
    std::size_t atomCount = 0;
    std::size_t totalPoints = 0;

    for (const excgrid::Block& block : engine.Grid().Blocks())
    {
        if (block.points.size() != block.pointCount || block.weights.size() != block.pointCount ||
            block.atomIndex.size() != block.pointCount)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "EvaluateXcKernelDerivativeBlock: a block's points, weights and "
                           "owning atoms must each hold its point count entries"});
        }

        totalPoints += block.pointCount;

        for (const std::size_t atom : block.atomIndex)
        {
            atomCount = std::max(atomCount, atom + 1);
        }
    }

    if (totalPoints == 0 || atomCount == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "EvaluateXcKernelDerivativeBlock: the grid carries no points"});
    }

    const std::size_t nuclearCoordinateCount = 3 * atomCount;

    const Eigen::MatrixXd spinDensity = 0.5 * density;
    const auto weights =
        ShellDensityWeights(envelopes, internal::FlattenRowMajor(spinDensity), aoCount);

    if (!weights.has_value())
    {
        return std::unexpected(weights.error());
    }

    const std::vector<double>& shellWeights = *weights;
    XcKernelDerivativeBlock block;
    block.numOccupied = numOccupied;
    block.numVirtual = numVirtual;
    block.nuclearCoordinateCount = nuclearCoordinateCount;
    block.values.assign(nuclearCoordinateCount * numOccupied * numVirtual, 0.0);

    excgrid::DerivativeRequest request;
    request.parts = BothParts();
    request.order = excgrid::DerivativeOrder::kFirst;
    request.nuclearCoordinateCount = nuclearCoordinateCount;

    // The kernel's contraction capacity is over active components; a chunk of
    // displacement sides fills it.
    const std::size_t sides =
        std::max<std::size_t>(1, excgrid::kSecondDerivativeCapacity / activeCount);

    std::vector<double> values(aoCount, 0.0);
    std::vector<double> gradients(3 * aoCount, 0.0);
    std::vector<double> hessians(6 * aoCount, 0.0);
    std::vector<double> valueContraction(aoCount, 0.0);
    std::vector<double> gradientContraction(3 * aoCount, 0.0);
    std::vector<double> expandedHessian(9 * aoCount, 0.0);

    std::vector<double> aoValueDelta(aoCount, 0.0);
    std::vector<double> aoGradientDelta(3 * aoCount, 0.0);
    std::vector<double> aoValueDeltaContraction(aoCount, 0.0);
    std::vector<double> aoGradientDeltaContraction(3 * aoCount, 0.0);

    std::vector<double> moValue(numOrbitals, 0.0);
    std::vector<double> moGradient(3 * numOrbitals, 0.0);
    std::vector<double> moHessian(9 * numOrbitals, 0.0);

    std::vector<double> rightHandSide(sides * activeCount, 0.0);
    std::vector<double> displacementRho(sides, 0.0);
    std::vector<double> displacementGradient(sides * 3, 0.0);
    std::vector<double> displacementTau(sides, 0.0);
    std::vector<double> displacementMoValue(sides * numOrbitals, 0.0);
    std::vector<double> displacementMoGradient(sides * 3 * numOrbitals, 0.0);

    // Per orbital, per spin: G . grad(m) and its two displacement pieces.
    std::vector<double> gradientDotMoA(numOrbitals, 0.0);
    std::vector<double> deltaGradientDotMoA(numOrbitals, 0.0);
    std::vector<double> gradientDotDeltaMoA(numOrbitals, 0.0);
    std::vector<double> gradientDotMoB(numOrbitals, 0.0);
    std::vector<double> deltaGradientDotMoB(numOrbitals, 0.0);
    std::vector<double> gradientDotDeltaMoB(numOrbitals, 0.0);

    std::vector<std::size_t> selectedShells;
    std::vector<std::size_t> selectedAos;

    std::vector<double> positionFirst;
    std::vector<double> weightFirst;

    excgrid::PointInputs inputs;
    excgrid::PointResult pointResult;
    excgrid::PointSecondDerivative second;

    for (const excgrid::Block& gridBlock : engine.Grid().Blocks())
    {
        const excgrid::BlockDerivativeSizes sizes = excgrid::SizeOf(request, gridBlock.pointCount);
        positionFirst.resize(sizes.positionFirst);
        weightFirst.resize(sizes.weightFirst);

        excgrid::BlockDerivatives derivatives;
        derivatives.positionFirst = std::span<double>(positionFirst.data(), sizes.positionFirst);
        derivatives.weightFirst = std::span<double>(weightFirst.data(), sizes.weightFirst);
        derivatives.atomIndex = gridBlock.atomIndex;

        const excgrid::DerivativeStatus status = provider.Evaluate(gridBlock, request, derivatives);

        if (status != excgrid::DerivativeStatus::kOk)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "EvaluateXcKernelDerivativeBlock: the grid derivative provider "
                           "refused a block: " +
                               std::string(DerivativeStatusName(status))});
        }

        for (std::size_t p = 0; p < gridBlock.pointCount; ++p)
        {
            const double weight = gridBlock.weights[p];

            if (weight == 0.0)
            {
                continue;
            }

            selectedShells.clear();
            selectedAos.clear();

            for (std::size_t shell = 0; shell < envelopes.size(); ++shell)
            {
                const ShellEnvelope& envelope = envelopes[shell];

                if (!ShellIsSignificant(envelope,
                                        gridBlock.points[p],
                                        shellWeights[shell],
                                        settings.screeningTolerance))
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

            engine.Evaluator().EvaluateDerivativesSelected(
                gridBlock.points[p], selectedShells, values, gradients, hessians);

            const PointDensity field = ContractSelectedSpinDensity(
                spinDensity, values, gradients, selectedAos, valueContraction, gradientContraction);

            const double sigma = field.gradient[0] * field.gradient[0] +
                                 field.gradient[1] * field.gradient[1] +
                                 field.gradient[2] * field.gradient[2];

            // The orbital values and derivatives over the kept AOs, and each
            // AO's Hessian expanded once for the displacement walk.
            for (std::size_t orbital = 0; orbital < numOrbitals; ++orbital)
            {
                moValue[orbital] = 0.0;

                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    moGradient[3 * orbital + axis] = 0.0;
                }

                for (std::size_t entry = 0; entry < 9; ++entry)
                {
                    moHessian[9 * orbital + entry] = 0.0;
                }
            }

            for (const std::size_t mu : selectedAos)
            {
                const std::array<double, 9> hessian = ExpandHessian(hessians, mu);

                for (std::size_t entry = 0; entry < 9; ++entry)
                {
                    expandedHessian[9 * mu + entry] = hessian[entry];
                }

                for (std::size_t orbital = 0; orbital < numOrbitals; ++orbital)
                {
                    const double coefficient = moCoefficients(static_cast<Eigen::Index>(mu),
                                                              static_cast<Eigen::Index>(orbital));

                    moValue[orbital] += coefficient * values[mu];

                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        moGradient[3 * orbital + axis] += coefficient * gradients[3 * mu + axis];
                    }

                    for (std::size_t entry = 0; entry < 9; ++entry)
                    {
                        moHessian[9 * orbital + entry] += coefficient * hessian[entry];
                    }
                }
            }

            inputs = excgrid::PointInputs{};

            for (const excgrid::Component id : active)
            {
                inputs.Set(id, ComponentValue(id, field, sigma));
            }

            for (std::size_t start = 0; start < nuclearCoordinateCount; start += sides)
            {
                const std::size_t count = std::min(sides, nuclearCoordinateCount - start);

                for (std::size_t side = 0; side < count; ++side)
                {
                    const std::size_t coordinate = start + side;
                    const std::size_t atom = coordinate / 3;
                    const std::size_t direction = coordinate % 3;

                    // The two AO-level channels: the point moves through the
                    // displacement field, and an AO centered on the displaced
                    // atom moves against it.
                    for (const std::size_t mu : selectedAos)
                    {
                        const std::size_t base = (p * 3) * nuclearCoordinateCount + coordinate;
                        const double positionX = derivatives.positionFirst[base];
                        const double positionY =
                            derivatives.positionFirst[base + nuclearCoordinateCount];
                        const double positionZ =
                            derivatives.positionFirst[base + 2 * nuclearCoordinateCount];

                        double deltaValue = gradients[3 * mu + 0] * positionX +
                                            gradients[3 * mu + 1] * positionY +
                                            gradients[3 * mu + 2] * positionZ;

                        if (atomOfAo[mu] == atom)
                        {
                            deltaValue -= gradients[3 * mu + direction];
                        }

                        aoValueDelta[mu] = deltaValue;

                        for (std::size_t axis = 0; axis < 3; ++axis)
                        {
                            const double hessianX =
                                expandedHessian[9 * mu + 3 * axis + 0] * positionX +
                                expandedHessian[9 * mu + 3 * axis + 1] * positionY +
                                expandedHessian[9 * mu + 3 * axis + 2] * positionZ;

                            double deltaGradientAxis = hessianX;

                            if (atomOfAo[mu] == atom)
                            {
                                deltaGradientAxis -= expandedHessian[9 * mu + 3 * axis + direction];
                            }

                            aoGradientDelta[3 * mu + axis] = deltaGradientAxis;
                        }
                    }

                    for (const std::size_t mu : selectedAos)
                    {
                        double contractedValue = 0.0;
                        std::array<double, 3> contractedGradient{};

                        for (const std::size_t nu : selectedAos)
                        {
                            const double element = spinDensity(static_cast<Eigen::Index>(mu),
                                                               static_cast<Eigen::Index>(nu));

                            contractedValue += element * aoValueDelta[nu];

                            for (std::size_t axis = 0; axis < 3; ++axis)
                            {
                                contractedGradient[axis] +=
                                    element * aoGradientDelta[3 * nu + axis];
                            }
                        }

                        aoValueDeltaContraction[mu] = contractedValue;

                        for (std::size_t axis = 0; axis < 3; ++axis)
                        {
                            aoGradientDeltaContraction[3 * mu + axis] = contractedGradient[axis];
                        }
                    }

                    double deltaRho = 0.0;
                    std::array<double, 3> deltaGradient{};
                    double deltaTau = 0.0;

                    for (const std::size_t mu : selectedAos)
                    {
                        deltaRho += aoValueDelta[mu] * valueContraction[mu] +
                                    values[mu] * aoValueDeltaContraction[mu];

                        for (std::size_t axis = 0; axis < 3; ++axis)
                        {
                            deltaGradient[axis] +=
                                aoGradientDelta[3 * mu + axis] * valueContraction[mu] +
                                gradients[3 * mu + axis] * aoValueDeltaContraction[mu] +
                                aoValueDelta[mu] * gradientContraction[3 * mu + axis] +
                                values[mu] * aoGradientDeltaContraction[3 * mu + axis];

                            deltaTau += 0.5 * (aoGradientDelta[3 * mu + axis] *
                                                   gradientContraction[3 * mu + axis] +
                                               gradients[3 * mu + axis] *
                                                   aoGradientDeltaContraction[3 * mu + axis]);
                        }
                    }

                    displacementRho[side] = deltaRho;

                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        displacementGradient[3 * side + axis] = deltaGradient[axis];
                    }

                    displacementTau[side] = deltaTau;

                    for (std::size_t orbital = 0; orbital < numOrbitals; ++orbital)
                    {
                        double deltaValue = 0.0;
                        std::array<double, 3> deltaGradientOrbital{};

                        for (const std::size_t mu : selectedAos)
                        {
                            const double coefficient = moCoefficients(
                                static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(orbital));

                            deltaValue += coefficient * aoValueDelta[mu];

                            for (std::size_t axis = 0; axis < 3; ++axis)
                            {
                                deltaGradientOrbital[axis] +=
                                    coefficient * aoGradientDelta[3 * mu + axis];
                            }
                        }

                        displacementMoValue[side * numOrbitals + orbital] = deltaValue;

                        for (std::size_t axis = 0; axis < 3; ++axis)
                        {
                            displacementMoGradient[(side * 3 + axis) * numOrbitals + orbital] =
                                deltaGradientOrbital[axis];
                        }
                    }

                    // sigma_ss = g_s . g_s and sigma_ab = g_A . g_B, which the
                    // closed shell makes one field, so every sigma component
                    // moves by 2 g . delta(g).
                    const double gradientSigma = 2.0 * (field.gradient[0] * deltaGradient[0] +
                                                        field.gradient[1] * deltaGradient[1] +
                                                        field.gradient[2] * deltaGradient[2]);

                    for (std::size_t slot = 0; slot < activeCount; ++slot)
                    {
                        rightHandSide[side * activeCount + slot] =
                            ResponseComponent(active[slot], deltaRho, gradientSigma, deltaTau);
                    }
                }

                const excgrid::KernelStatus kernelStatus =
                    functional.EvaluatePointWithSecondDerivatives(
                        inputs,
                        std::span<const double>(rightHandSide.data(), count * activeCount),
                        pointResult,
                        second);

                if (kernelStatus != excgrid::KernelStatus::kOk)
                {
                    return std::unexpected(
                        qcx::Error{qcx::ErrorCode::kInternalError,
                                   "EvaluateXcKernelDerivativeBlock: the kernel refused a "
                                   "contraction: " +
                                       std::string(excgrid::DescribeStatus(kernelStatus))});
                }

                const auto first = [&pointResult](excgrid::Component id) {
                    return pointResult.mask.Test(id) ? pointResult.first[excgrid::IndexOf(id)]
                                                     : 0.0;
                };

                const double vrhoA = first(excgrid::Component::RhoA);
                const double vrhoB = first(excgrid::Component::RhoB);
                const double vSigmaAa = first(excgrid::Component::SigmaAa);
                const double vSigmaAb = first(excgrid::Component::SigmaAb);
                const double vSigmaBb = first(excgrid::Component::SigmaBb);
                const double vTauA = first(excgrid::Component::TauA);
                const double vTauB = first(excgrid::Component::TauB);

                // G_s = 2 v_sigma,ss g_s + v_sigma,ab g_other; in the closed
                // shell both spin gradients are the one field gradient.
                const double gradientScaleA = 2.0 * vSigmaAa + vSigmaAb;
                const double gradientScaleB = 2.0 * vSigmaBb + vSigmaAb;

                for (std::size_t side = 0; side < count; ++side)
                {
                    const std::size_t coordinate = start + side;

                    double deltaVrhoA = 0.0;
                    double deltaVrhoB = 0.0;
                    double deltaVSigmaAa = 0.0;
                    double deltaVSigmaAb = 0.0;
                    double deltaVSigmaBb = 0.0;
                    double deltaVTauA = 0.0;
                    double deltaVTauB = 0.0;

                    for (std::size_t slot = 0; slot < activeCount; ++slot)
                    {
                        const double contracted = second.contracted[side * activeCount + slot];

                        switch (active[slot])
                        {
                        case excgrid::Component::RhoA:
                            deltaVrhoA = contracted;
                            break;
                        case excgrid::Component::RhoB:
                            deltaVrhoB = contracted;
                            break;
                        case excgrid::Component::SigmaAa:
                            deltaVSigmaAa = contracted;
                            break;
                        case excgrid::Component::SigmaAb:
                            deltaVSigmaAb = contracted;
                            break;
                        case excgrid::Component::SigmaBb:
                            deltaVSigmaBb = contracted;
                            break;
                        case excgrid::Component::TauA:
                            deltaVTauA = contracted;
                            break;
                        case excgrid::Component::TauB:
                            deltaVTauB = contracted;
                            break;
                        default:
                            break;
                        }
                    }

                    std::array<double, 3> deltaGradientA{};
                    std::array<double, 3> deltaGradientB{};

                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        deltaGradientA[axis] =
                            gradientScaleA * displacementGradient[3 * side + axis] +
                            (2.0 * deltaVSigmaAa + deltaVSigmaAb) * field.gradient[axis];
                        deltaGradientB[axis] =
                            gradientScaleB * displacementGradient[3 * side + axis] +
                            (2.0 * deltaVSigmaBb + deltaVSigmaAb) * field.gradient[axis];
                    }

                    // The per-orbital pieces the bracket and its displacement
                    // derivative share.
                    for (std::size_t orbital = 0; orbital < numOrbitals; ++orbital)
                    {
                        const std::size_t gradientBase = 3 * orbital;
                        const std::size_t deltaBase = side * 3 * numOrbitals + orbital;

                        const double fieldDotGradient =
                            field.gradient[0] * moGradient[gradientBase + 0] +
                            field.gradient[1] * moGradient[gradientBase + 1] +
                            field.gradient[2] * moGradient[gradientBase + 2];

                        const std::array<double, 3> deltaMoGradientOrbital = {
                            displacementMoGradient[deltaBase],
                            displacementMoGradient[deltaBase + numOrbitals],
                            displacementMoGradient[deltaBase + 2 * numOrbitals]};

                        gradientDotMoA[orbital] = gradientScaleA * fieldDotGradient;
                        gradientDotMoB[orbital] = gradientScaleB * fieldDotGradient;

                        deltaGradientDotMoA[orbital] =
                            deltaGradientA[0] * moGradient[gradientBase + 0] +
                            deltaGradientA[1] * moGradient[gradientBase + 1] +
                            deltaGradientA[2] * moGradient[gradientBase + 2];
                        deltaGradientDotMoB[orbital] =
                            deltaGradientB[0] * moGradient[gradientBase + 0] +
                            deltaGradientB[1] * moGradient[gradientBase + 1] +
                            deltaGradientB[2] * moGradient[gradientBase + 2];

                        gradientDotDeltaMoA[orbital] =
                            gradientScaleA * (field.gradient[0] * deltaMoGradientOrbital[0] +
                                              field.gradient[1] * deltaMoGradientOrbital[1] +
                                              field.gradient[2] * deltaMoGradientOrbital[2]);
                        gradientDotDeltaMoB[orbital] = gradientDotDeltaMoA[orbital];
                    }

                    const double weightDerivative =
                        derivatives.weightFirst[p * nuclearCoordinateCount + coordinate];

                    for (std::size_t occupied = 0; occupied < numOccupied; ++occupied)
                    {
                        for (std::size_t virtualOrbital = 0; virtualOrbital < numVirtual;
                             ++virtualOrbital)
                        {
                            const std::size_t i = occupied;
                            const std::size_t a = numOccupied + virtualOrbital;

                            const double valueI = moValue[i];
                            const double valueA = moValue[a];
                            const double deltaValueI = displacementMoValue[side * numOrbitals + i];
                            const double deltaValueA = displacementMoValue[side * numOrbitals + a];

                            double tauPair = 0.0;
                            double deltaTauPair = 0.0;

                            for (std::size_t axis = 0; axis < 3; ++axis)
                            {
                                tauPair += moGradient[3 * a + axis] * moGradient[3 * i + axis];
                                deltaTauPair +=
                                    displacementMoGradient[(side * 3 + axis) * numOrbitals + a] *
                                        moGradient[3 * i + axis] +
                                    moGradient[3 * a + axis] *
                                        displacementMoGradient[(side * 3 + axis) * numOrbitals + i];
                            }

                            const double bracketA =
                                vrhoA * valueA * valueI + gradientDotMoA[a] * valueI +
                                valueA * gradientDotMoA[i] + 0.5 * vTauA * tauPair;
                            const double bracketB =
                                vrhoB * valueA * valueI + gradientDotMoB[a] * valueI +
                                valueA * gradientDotMoB[i] + 0.5 * vTauB * tauPair;

                            const double deltaBracketA =
                                deltaVrhoA * valueA * valueI +
                                vrhoA * (deltaValueA * valueI + valueA * deltaValueI) +
                                (deltaGradientDotMoA[a] + gradientDotDeltaMoA[a]) * valueI +
                                gradientDotMoA[a] * deltaValueI + deltaValueA * gradientDotMoA[i] +
                                valueA * (deltaGradientDotMoA[i] + gradientDotDeltaMoA[i]) +
                                0.5 * deltaVTauA * tauPair + 0.5 * vTauA * deltaTauPair;
                            const double deltaBracketB =
                                deltaVrhoB * valueA * valueI +
                                vrhoB * (deltaValueA * valueI + valueA * deltaValueI) +
                                (deltaGradientDotMoB[a] + gradientDotDeltaMoB[a]) * valueI +
                                gradientDotMoB[a] * deltaValueI + deltaValueA * gradientDotMoB[i] +
                                valueA * (deltaGradientDotMoB[i] + gradientDotDeltaMoB[i]) +
                                0.5 * deltaVTauB * tauPair + 0.5 * vTauB * deltaTauPair;

                            block.values[coordinate * numOccupied * numVirtual +
                                         occupied * numVirtual + virtualOrbital] +=
                                0.5 * (weightDerivative * bracketA + weight * deltaBracketA) +
                                0.5 * (weightDerivative * bracketB + weight * deltaBracketB);
                        }
                    }
                }
            }
        }
    }

    return block;
}

} // namespace qcx::grid
