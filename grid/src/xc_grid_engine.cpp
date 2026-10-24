#include "qcx/grid/xc_grid_engine.hpp"

#include "internal/xc_point_assembly.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/grid/shell_screening.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::grid {

namespace {

// The registry's names, comma-separated, for the unknown-name message.
std::string ShippedFunctionalNames() {
    std::string names;

    for (const std::string_view name : excgrid::FunctionalNames())
    {
        if (!names.empty())
        {
            names += ", ";
        }

        names += name;
    }

    return names;
}

// The components a mask names, comma-separated, for the refusal message.
std::string ComponentNames(const excgrid::ComponentMask& mask) {
    std::string names;

    for (std::size_t index = 0; index < excgrid::kComponentCapacity; ++index)
    {
        const auto id = static_cast<excgrid::Component>(index);

        if (!mask.Test(id))
        {
            continue;
        }

        if (!names.empty())
        {
            names += ", ";
        }

        names += excgrid::Describe(id).name;
    }

    return names;
}

// The per-point inputs this engine always supplies: both densities, the three
// gradient invariants, and both kinetic-energy densities. A functional whose
// RequiredMask() names anything else cannot be integrated here.
constexpr excgrid::ComponentMask kSuppliedComponents = excgrid::ComponentMask::Collinear();

// One spin's density and density gradient at a grid point.
using PointDensity = internal::PointDensity;

// Contracts a spin density matrix with the point's AO values and gradients:
//     rho      = sum_mu nu D_mu nu phi_mu phi_nu
//     grad rho = sum_mu nu D_mu nu (grad phi_mu phi_nu + phi_mu grad phi_nu)
//     tau      = 1/2 sum_mu nu D_mu nu grad phi_mu . grad phi_nu
// All three sums factor through (D phi)_mu and (D grad phi)_mu, so one point
// costs O(nAO^2) rather than O(nAO^3). The double-sum form is written out, so
// the result is exact for a non-symmetric D as well.
PointDensity ContractSpinDensity(const Eigen::MatrixXd& density,
                                 const std::vector<double>& values,
                                 const std::vector<double>& gradients,
                                 std::vector<double>& valueContraction,
                                 std::vector<double>& gradientContraction) {
    const Eigen::Index n = density.rows();
    const Eigen::Map<const Eigen::VectorXd> phi(values.data(), n);
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> gradient(
        gradients.data(), n, 3);
    Eigen::Map<Eigen::VectorXd> densityPhi(valueContraction.data(), n);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> densityGradient(
        gradientContraction.data(), n, 3);

    densityPhi.noalias() = density * phi;
    densityGradient.noalias() = density * gradient;

    PointDensity result;
    result.rho = phi.dot(densityPhi);

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const Eigen::Index index = static_cast<Eigen::Index>(axis);
        result.gradient[axis] =
            gradient.col(index).dot(densityPhi) + phi.dot(densityGradient.col(index));
        result.tau += 0.5 * gradient.col(index).dot(densityGradient.col(index));
    }

    return result;
}

// Accumulates one spin's potential contribution at a point:
//     V_mu nu += w [ vrho phi_mu phi_nu
//                    + G . (grad phi_mu phi_nu + phi_mu grad phi_nu)
//                    + 1/2 vtau grad phi_mu . grad phi_nu ]
// written as two rank-1 updates plus, when the functional reads tau, one
// rank-3 outer product, with G the sigma-weighted density gradient the kernel
// returned. The form is symmetric in mu and nu by construction, and the
// O(nAO^2) outer products are the only per-point cost. A zero vtau skips the
// tau product, so an LDA/GGA potential keeps its bits.
void AccumulatePotential(Eigen::MatrixXd& potential,
                         const std::vector<double>& values,
                         const std::vector<double>& gradients,
                         // (vrho, vtau) are the two density-component derivatives, in the
                         // kernel's order; the weight is not interchangeable with either.
                         // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                         double vrho,
                         double vtau,
                         const Eigen::Vector3d& gradientWeight,
                         double weight) {
    const Eigen::Index n = potential.rows();
    const Eigen::Map<const Eigen::VectorXd> phi(values.data(), n);
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> gradient(
        gradients.data(), n, 3);
    const Eigen::VectorXd vrhoPhi = vrho * phi + gradient * gradientWeight;
    const Eigen::VectorXd gradientPhi = gradient * gradientWeight;

    potential.noalias() += weight * (vrhoPhi * phi.transpose() + phi * gradientPhi.transpose());

    if (vtau != 0.0)
    {
        potential.noalias() += (weight * 0.5 * vtau) * gradient * gradient.transpose();
    }
}

// Contracts a spin density matrix with the point's AO values alone:
//     rho = sum_mu nu D_mu nu phi_mu phi_nu
// The values-only tier of the contraction, for a functional whose kernel
// ignores the density gradients: the sigma inputs stay zero and the potential
// accumulation below degenerates to its vrho term.
double ContractValue(const Eigen::MatrixXd& density,
                     const std::vector<double>& values,
                     std::vector<double>& valueContraction) {
    const Eigen::Index n = density.rows();
    const Eigen::Map<const Eigen::VectorXd> phi(values.data(), n);
    Eigen::Map<Eigen::VectorXd> densityPhi(valueContraction.data(), n);

    densityPhi.noalias() = density * phi;

    return phi.dot(densityPhi);
}

// One point's kernel call, by named component rather than by position: the
// positional adapter cannot carry tau and reports a refusal as zeros. Every
// component the engine supplies is set, so the input mask is the seven-bit
// collinear mask; the result's named first derivatives are read back, and a
// slot the functional did not produce stays at its initial zero. A refusal is
// returned as an Error naming the status, never as a silent zero energy.
qcx::Result<excgrid::XcKernelValue> EvaluateKernel(
    const excgrid::XcFunctional& functional,
    std::string_view operation,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const PointDensity& densityA,
    const PointDensity& densityB,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const Eigen::Vector3d& gradientA,
    const Eigen::Vector3d& gradientB) {
    excgrid::PointInputs inputs;
    inputs.Set(excgrid::Component::RhoA, densityA.rho);
    inputs.Set(excgrid::Component::RhoB, densityB.rho);
    inputs.Set(excgrid::Component::SigmaAa, gradientA.dot(gradientA));
    inputs.Set(excgrid::Component::SigmaAb, gradientA.dot(gradientB));
    inputs.Set(excgrid::Component::SigmaBb, gradientB.dot(gradientB));
    inputs.Set(excgrid::Component::TauA, densityA.tau);
    inputs.Set(excgrid::Component::TauB, densityB.tau);

    excgrid::PointResult result;
    const excgrid::KernelStatus status = functional.EvaluatePoint(inputs, result);

    if (status != excgrid::KernelStatus::kOk)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                          std::string(operation) + ": the functional " +
                                              std::string(excgrid::DescribeStatus(status))});
    }

    excgrid::XcKernelValue kernel;
    kernel.exc = result.exc;
    kernel.vrhoA = result.first[excgrid::IndexOf(excgrid::Component::RhoA)];
    kernel.vrhoB = result.first[excgrid::IndexOf(excgrid::Component::RhoB)];
    kernel.vsigmaAa = result.first[excgrid::IndexOf(excgrid::Component::SigmaAa)];
    kernel.vsigmaAb = result.first[excgrid::IndexOf(excgrid::Component::SigmaAb)];
    kernel.vsigmaBb = result.first[excgrid::IndexOf(excgrid::Component::SigmaBb)];
    kernel.vtauA = result.first[excgrid::IndexOf(excgrid::Component::TauA)];
    kernel.vtauB = result.first[excgrid::IndexOf(excgrid::Component::TauB)];

    return kernel;
}

// The screened form of AccumulatePotential, on the same selected block:
//     V_mu nu += w [ (vrho phi_mu + G . grad phi_mu) phi_nu
//                    + phi_mu (G . grad phi_nu)
//                    + 1/2 vtau grad phi_mu . grad phi_nu ]
// The dense form's two rank-1 updates become one (mu, nu) walk over the same
// terms, with the second term's dot product taken on NU - taking it on mu
// instead is symmetric only when it is zero, which is why an LDA functional
// cannot tell the difference and a GGA one can. A functional that ignores the
// gradients passes a zero gradient weight and a zero-filled gradient buffer,
// which reduces this to the vrho term exactly as the dense path reduces its
// own; a zero vtau likewise skips the tau term and keeps an LDA/GGA result
// bit-identical. The per-AO dot products are hoisted into a caller-owned
// scratch buffer, so the pair loop costs one multiply-add per term plus three
// for the tau contraction.
void AccumulateSelectedPotential(
    Eigen::MatrixXd& potential,
    // (values, gradients) is the value-then-derivative order the accumulators are addressed by.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    std::span<const double> values,
    std::span<const double> gradients,
    // (vrho, vtau) are the two density-component derivatives, in the kernel's order; the
    // weight is not interchangeable with either.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    double vrho,
    double vtau,
    const Eigen::Vector3d& gradientWeight,
    double weight,
    std::span<const std::size_t> selectedAos,
    std::vector<double>& gradientWeightedValues) {
    for (const std::size_t mu : selectedAos)
    {
        gradientWeightedValues[mu] = gradients[3 * mu + 0] * gradientWeight[0] +
                                     gradients[3 * mu + 1] * gradientWeight[1] +
                                     gradients[3 * mu + 2] * gradientWeight[2];
    }

    const bool usesTau = vtau != 0.0;

    for (const std::size_t mu : selectedAos)
    {
        const double valueMu = values[mu];
        const double vrhoPhi = vrho * valueMu + gradientWeightedValues[mu];

        for (const std::size_t nu : selectedAos)
        {
            double term = vrhoPhi * values[nu] + valueMu * gradientWeightedValues[nu];

            if (usesTau)
            {
                double gradientDot = 0.0;

                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    gradientDot += gradients[3 * mu + axis] * gradients[3 * nu + axis];
                }

                term += 0.5 * vtau * gradientDot;
            }

            potential(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) +=
                weight * term;
        }
    }
}

} // namespace

XcGridEngine::XcGridEngine(excgrid::BlockGrid grid,
                           const excgrid::XcFunctional* functional,
                           AoEvaluator evaluator,
                           std::vector<ShellEnvelope> envelopes) noexcept :
    _grid(std::move(grid)), _functional(functional), _evaluator(std::move(evaluator)),
    _envelopes(std::move(envelopes)), _aoCount(_evaluator.AOCount()) {}

qcx::Result<XcGridEngine> XcGridEngine::Create(const qcx::molecule::Molecule& molecule,
                                               const qcx::basisset::BasisSet& basis,
                                               std::string_view functionalName,
                                               const XcGridSettings& settings) {
    const excgrid::XcFunctional* functional = excgrid::FindFunctional(functionalName);

    if (functional == nullptr)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "XcGridEngine::Create: unknown XC functional '" +
                                              std::string(functionalName) +
                                              "'; shipped names: " + ShippedFunctionalNames()});
    }

    // The refusal is keyed on the functional's declared inputs, not on its
    // name, so the next functional that reaches past the engine's components
    // is caught at creation rather than reading a silent zero at every point.
    const excgrid::ComponentMask unsupplied{functional->RequiredMask().bits &
                                            ~kSuppliedComponents.bits};

    if (unsupplied.bits != 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented,
                       "XcGridEngine::Create: functional '" + std::string(functionalName) +
                           "' requires components this engine does not supply: " +
                           ComponentNames(unsupplied)});
    }

    auto geometry = ToExcgridGeometry(molecule);

    if (!geometry.has_value())
    {
        return std::unexpected(geometry.error());
    }

    auto grid = excgrid::BlockGrid::Create(*geometry, ToExcgridParams(settings));

    if (!grid.has_value())
    {
        return std::unexpected(
            TranslateExcgridError(grid.error(), "XcGridEngine::Create: block grid"));
    }

    auto evaluator = AoEvaluator::Create(molecule, basis);

    if (!evaluator.has_value())
    {
        return std::unexpected(evaluator.error());
    }

    // The screening envelopes are density-independent, so they are built once
    // here and reused by every screened integration. The builder checks the
    // ranges against the basis rather than trusting that the two orders agree:
    // a mismatch would screen the wrong shells and fail silently.
    auto envelopes = BuildShellEnvelopes(molecule, basis, evaluator->ShellRanges());

    if (!envelopes.has_value())
    {
        return std::unexpected(envelopes.error());
    }

    return XcGridEngine(std::move(*grid), functional, std::move(*evaluator), std::move(*envelopes));
}

qcx::Result<XcEvaluation> XcGridEngine::Evaluate(const Eigen::MatrixXd& densityAlpha,
                                                 const Eigen::MatrixXd& densityBeta) const {
    const Eigen::Index n = static_cast<Eigen::Index>(_aoCount);

    if (densityAlpha.rows() != n || densityAlpha.cols() != n || densityBeta.rows() != n ||
        densityBeta.cols() != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "XcGridEngine::Evaluate: both densities must be AOCount() x AOCount()"});
    }

    XcEvaluation evaluation;
    evaluation.potentialAlpha = Eigen::MatrixXd::Zero(n, n);
    evaluation.potentialBeta = Eigen::MatrixXd::Zero(n, n);

    // Per-point scratch, reused across the whole grid. No hessian block: the
    // engine consumes values and first derivatives only, and the AO tier's
    // gradient entry point is what keeps the second derivatives out of the
    // per-point cost.
    std::vector<double> values(_aoCount);
    std::vector<double> gradients(3 * _aoCount);
    std::vector<double> valueContraction(_aoCount);
    std::vector<double> gradientContraction(3 * _aoCount);

    // A functional whose kernel ignores sigma AND tau (every LDA name) needs
    // neither the AO gradients nor the gradient contraction, so the whole
    // gradient tier is skipped for it rather than computed and discarded.
    // Tau needs the AO gradients even when sigma is not read, so it is part
    // of the gate rather than riding on UsesGradient alone.
    const excgrid::ComponentMask required = _functional->RequiredMask();
    const bool usesGradient = _functional->UsesGradient() ||
                              required.Test(excgrid::Component::TauA) ||
                              required.Test(excgrid::Component::TauB);

    for (const excgrid::Block& block : _grid.Blocks())
    {
        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            const double weight = block.weights[point];

            if (weight == 0.0)
            {
                continue;
            }

            ++evaluation.counts.points;

            if (usesGradient)
            {
                _evaluator.EvaluateGradients(block.points[point], values, gradients);
            } else
            {
                _evaluator.Evaluate(block.points[point], values);
            }

            PointDensity densityA;
            PointDensity densityB;

            if (usesGradient)
            {
                densityA = ContractSpinDensity(
                    densityAlpha, values, gradients, valueContraction, gradientContraction);
                densityB = ContractSpinDensity(
                    densityBeta, values, gradients, valueContraction, gradientContraction);
            } else
            {
                // The density gradients and tau stay zero, which is what the
                // kernel ignores anyway: the accumulation below then reduces
                // to its vrho term without a special case.
                densityA.rho = ContractValue(densityAlpha, values, valueContraction);
                densityB.rho = ContractValue(densityBeta, values, valueContraction);
            }

            const Eigen::Vector3d gradientA(
                densityA.gradient[0], densityA.gradient[1], densityA.gradient[2]);
            const Eigen::Vector3d gradientB(
                densityB.gradient[0], densityB.gradient[1], densityB.gradient[2]);

            const auto kernel = EvaluateKernel(
                *_functional, "XcGridEngine::Evaluate", densityA, densityB, gradientA, gradientB);

            if (!kernel.has_value())
            {
                return std::unexpected(kernel.error());
            }

            evaluation.energy += weight * kernel->exc;

            AccumulatePotential(evaluation.potentialAlpha,
                                values,
                                gradients,
                                kernel->vrhoA,
                                kernel->vtauA,
                                2.0 * kernel->vsigmaAa * gradientA + kernel->vsigmaAb * gradientB,
                                weight);
            AccumulatePotential(evaluation.potentialBeta,
                                values,
                                gradients,
                                kernel->vrhoB,
                                kernel->vtauB,
                                2.0 * kernel->vsigmaBb * gradientB + kernel->vsigmaAb * gradientA,
                                weight);
        }
    }

    // The dense path runs no significance tests and weighs no density: those
    // columns stay zero, and its own costs appear in both the actual and the
    // dense columns, which is what makes the two paths comparable.
    const std::size_t slots = evaluation.counts.points * _aoCount;
    evaluation.counts.aoSlotsEvaluated = slots;
    evaluation.counts.aoSlotsDense = slots;
    evaluation.counts.contractionPairs = 2 * slots * _aoCount;
    evaluation.counts.contractionPairsDense = evaluation.counts.contractionPairs;

    return evaluation;
}

qcx::Result<XcEvaluation> XcGridEngine::EvaluateScreened(const Eigen::MatrixXd& densityAlpha,
                                                         const Eigen::MatrixXd& densityBeta,
                                                         double tolerance,
                                                         const XcBatchSettings& batch) const {
    const Eigen::Index n = static_cast<Eigen::Index>(_aoCount);

    if (densityAlpha.rows() != n || densityAlpha.cols() != n || densityBeta.rows() != n ||
        densityBeta.cols() != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "XcGridEngine::EvaluateScreened: both densities must be AOCount() x "
                       "AOCount()"});
    }

    XcEvaluation evaluation;
    evaluation.potentialAlpha = Eigen::MatrixXd::Zero(n, n);
    evaluation.potentialBeta = Eigen::MatrixXd::Zero(n, n);

    // The per-point scratch the assembly reads and writes, one point at a time.
    std::vector<double> valueContraction(_aoCount);
    std::vector<double> gradientContraction(3 * _aoCount);
    std::vector<double> gradientWeightedValues(_aoCount);

    // The weights cost one pass over each spin's density matrix, O(AOCount()^2)
    // against the O(points x AOCount()^2) of the assembly: setup, but counted.
    // A closed-shell pair is one matrix, so its pass is shared.
    const std::vector<double> flatAlpha = internal::FlattenRowMajor(densityAlpha);
    auto weightsAlpha = ShellDensityWeights(_envelopes, flatAlpha, _aoCount);

    if (!weightsAlpha.has_value())
    {
        return std::unexpected(weightsAlpha.error());
    }

    const bool sharedSpinDensity = densityAlpha.data() == densityBeta.data();
    std::vector<double> weightsBeta;

    if (sharedSpinDensity)
    {
        weightsBeta = *weightsAlpha;
    } else
    {
        const std::vector<double> flatBeta = internal::FlattenRowMajor(densityBeta);
        auto computed = ShellDensityWeights(_envelopes, flatBeta, _aoCount);

        if (!computed.has_value())
        {
            return std::unexpected(computed.error());
        }

        weightsBeta = std::move(*computed);
    }

    evaluation.counts.densityWeightTerms = _aoCount * _aoCount * (sharedSpinDensity ? 1 : 2);

    // The batch buffers, sized once for the whole walk rather than per block:
    // one slice of AO values and one of AO gradients per batched point, plus
    // that point's selection in the two flat lists the fetch and the assembly
    // both walk - the surviving shells, and the AO slots they cover. The size
    // is clamped to the largest block, so a batch can never span blocks and a
    // large setting cannot allocate past one block's worth of points.
    std::size_t largestBlock = 0;

    for (const excgrid::Block& block : _grid.Blocks())
    {
        largestBlock = std::max(largestBlock, block.pointCount);
    }

    const std::size_t bufferPoints =
        batch.pointBatch == 0 ? largestBlock : std::min(batch.pointBatch, largestBlock);

    std::vector<double> batchValues(bufferPoints * _aoCount);
    std::vector<double> batchGradients(bufferPoints * 3 * _aoCount);
    std::vector<std::size_t> batchShells(bufferPoints * _envelopes.size());
    std::vector<std::size_t> batchAos(bufferPoints * _aoCount);
    std::vector<std::size_t> batchShellOffsets(bufferPoints + 1);
    std::vector<std::size_t> batchAoOffsets(bufferPoints + 1);
    std::vector<double> batchWeights(bufferPoints);

    // Tau needs the AO gradients even when sigma is not read, so the gate is
    // the functional's own gradient flag extended by the two tau bits.
    const excgrid::ComponentMask required = _functional->RequiredMask();
    const bool usesGradient = _functional->UsesGradient() ||
                              required.Test(excgrid::Component::TauA) ||
                              required.Test(excgrid::Component::TauB);

    for (const excgrid::Block& block : _grid.Blocks())
    {
        const std::size_t batchSize =
            batch.pointBatch == 0 ? block.pointCount : std::min(batch.pointBatch, block.pointCount);

        for (std::size_t start = 0; start < block.pointCount; start += batchSize)
        {
            const std::size_t count = std::min(batchSize, block.pointCount - start);
            ++evaluation.counts.batches;

            // Phase one of the batch: select and FETCH. Every point gets its AO
            // data here, once, into its own slice. A point that keeps no shell
            // gets an empty range and never enters the AO tier: its density is
            // below the tolerance everywhere, and the kernel's integrand is not
            // defined at rho = 0.
            std::size_t shellCursor = 0;
            std::size_t aoCursor = 0;

            for (std::size_t index = 0; index < count; ++index)
            {
                const std::size_t point = start + index;
                batchShellOffsets[index] = shellCursor;
                batchAoOffsets[index] = aoCursor;
                batchWeights[index] = block.weights[point];

                if (block.weights[point] == 0.0)
                {
                    continue;
                }

                ++evaluation.counts.points;

                for (std::size_t shell = 0; shell < _envelopes.size(); ++shell)
                {
                    ++evaluation.counts.shellTests;

                    // The stronger of the two spins decides: a shell that
                    // matters to neither spin can be dropped, and the neglected
                    // term in each spin is bounded by that spin's own weight.
                    const double spinWeight = std::max((*weightsAlpha)[shell], weightsBeta[shell]);

                    if (!ShellIsSignificant(
                            _envelopes[shell], block.points[point], spinWeight, tolerance))
                    {
                        continue;
                    }

                    ++evaluation.counts.shellKeeps;
                    batchShells[shellCursor++] = shell;

                    for (std::size_t ao = 0; ao < _envelopes[shell].functionCount; ++ao)
                    {
                        batchAos[aoCursor++] = _envelopes[shell].aoOffset + ao;
                    }
                }

                if (batchShellOffsets[index] == shellCursor)
                {
                    continue;
                }

                const std::size_t selectedCount = aoCursor - batchAoOffsets[index];
                evaluation.counts.aoSlotsEvaluated += selectedCount;
                evaluation.counts.contractionPairs += 2 * selectedCount * selectedCount;
                ++evaluation.counts.aoFetches;

                const std::span<double> values(batchValues.data() + index * _aoCount, _aoCount);
                const std::span<double> gradients(batchGradients.data() + index * 3 * _aoCount,
                                                  3 * _aoCount);
                const std::span<const std::size_t> selection(
                    batchShells.data() + batchShellOffsets[index],
                    shellCursor - batchShellOffsets[index]);

                if (usesGradient)
                {
                    _evaluator.EvaluateGradientsSelected(
                        block.points[point], selection, values, gradients);
                } else
                {
                    _evaluator.EvaluateSelected(block.points[point], selection, values);
                }
            }

            batchShellOffsets[count] = shellCursor;
            batchAoOffsets[count] = aoCursor;

            // Phase two: assemble. Every quantity of a point - the density, its
            // gradient, and both spin potentials - reads the slice phase one
            // filled, and none of them re-enters the AO tier.
            for (std::size_t index = 0; index < count; ++index)
            {
                if (batchShellOffsets[index] == batchShellOffsets[index + 1])
                {
                    continue;
                }

                const std::span<const double> values(batchValues.data() + index * _aoCount,
                                                     _aoCount);
                const std::span<const double> gradients(
                    batchGradients.data() + index * 3 * _aoCount, 3 * _aoCount);
                const std::span<const std::size_t> selectedAos(
                    batchAos.data() + batchAoOffsets[index],
                    batchAoOffsets[index + 1] - batchAoOffsets[index]);

                const PointDensity densityA =
                    internal::ContractSelectedSpinDensity(densityAlpha,
                                                          values,
                                                          gradients,
                                                          selectedAos,
                                                          valueContraction,
                                                          gradientContraction);
                const PointDensity densityB =
                    internal::ContractSelectedSpinDensity(densityBeta,
                                                          values,
                                                          gradients,
                                                          selectedAos,
                                                          valueContraction,
                                                          gradientContraction);

                const Eigen::Vector3d gradientA(
                    densityA.gradient[0], densityA.gradient[1], densityA.gradient[2]);
                const Eigen::Vector3d gradientB(
                    densityB.gradient[0], densityB.gradient[1], densityB.gradient[2]);

                const auto kernel = EvaluateKernel(*_functional,
                                                   "XcGridEngine::EvaluateScreened",
                                                   densityA,
                                                   densityB,
                                                   gradientA,
                                                   gradientB);

                if (!kernel.has_value())
                {
                    return std::unexpected(kernel.error());
                }

                const double weight = batchWeights[index];
                evaluation.energy += weight * kernel->exc;

                AccumulateSelectedPotential(evaluation.potentialAlpha,
                                            values,
                                            gradients,
                                            kernel->vrhoA,
                                            kernel->vtauA,
                                            2.0 * kernel->vsigmaAa * gradientA +
                                                kernel->vsigmaAb * gradientB,
                                            weight,
                                            selectedAos,
                                            gradientWeightedValues);
                AccumulateSelectedPotential(evaluation.potentialBeta,
                                            values,
                                            gradients,
                                            kernel->vrhoB,
                                            kernel->vtauB,
                                            2.0 * kernel->vsigmaBb * gradientB +
                                                kernel->vsigmaAb * gradientA,
                                            weight,
                                            selectedAos,
                                            gradientWeightedValues);
            }
        }
    }

    evaluation.counts.aoSlotsDense = evaluation.counts.points * _aoCount;
    evaluation.counts.contractionPairsDense = 2 * evaluation.counts.points * _aoCount * _aoCount;

    return evaluation;
}

qcx::Result<XcEvaluation> XcGridEngine::EvaluateClosedShell(const Eigen::MatrixXd& density) const {
    const Eigen::Index n = static_cast<Eigen::Index>(_aoCount);

    if (density.rows() != n || density.cols() != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "XcGridEngine::EvaluateClosedShell: density must be AOCount() x AOCount()"});
    }

    const Eigen::MatrixXd halfDensity = 0.5 * density;

    return Evaluate(halfDensity, halfDensity);
}

qcx::Result<XcEvaluation> XcGridEngine::EvaluateClosedShellScreened(
    const Eigen::MatrixXd& density, double tolerance, const XcBatchSettings& batch) const {
    const Eigen::Index n = static_cast<Eigen::Index>(_aoCount);

    if (density.rows() != n || density.cols() != n)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "XcGridEngine::EvaluateClosedShellScreened: density must be AOCount() x AOCount()"});
    }

    const Eigen::MatrixXd halfDensity = 0.5 * density;

    return EvaluateScreened(halfDensity, halfDensity, tolerance, batch);
}

} // namespace qcx::grid
