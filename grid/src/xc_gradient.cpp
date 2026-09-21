#include "qcx/grid/xc_gradient.hpp"

#include "internal/xc_point_assembly.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace qcx::grid {

namespace {

using internal::PointDensity;

// The evaluation's hessian order: xx, xy, xz, yy, yz, zz, mu-major.
constexpr std::array<std::array<std::size_t, 2>, 6> kHessianComponent = {
    {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}}};

// The slot value every AO is filled with before the envelopes claim it, so that
// an envelope set that does not cover the basis is refused rather than walked.
constexpr std::size_t kNoAtom = std::numeric_limits<std::size_t>::max();

// Both parts at once: DerivativePart names one bit each. The combination is not
// one of its enumerators, which is what a bitmask is for, and the analyzer reads
// the enumerator list rather than the underlying type as the cast's domain.
excgrid::DerivativePart BothParts() noexcept {
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    return static_cast<excgrid::DerivativePart>(
        static_cast<std::uint8_t>(excgrid::DerivativePart::kCoordinates) |
        static_cast<std::uint8_t>(excgrid::DerivativePart::kWeights));
}

// A provider refusal's name, for the error the walk returns instead of a
// gradient: a refused block answered with zeros is a wrong force that looks
// plausible, which is what the refusal exists to prevent.
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

// One spin's density field at a point: the value, its first derivative, and -
// for a functional that consumes the density gradients - the second.
struct PointField {
    double rho = 0.0;
    std::array<double, 3> gradient{};
    // Row-major symmetric 3x3, written only when the second derivative is asked
    // for: a functional that reads only rho and its gradient never needs it, and
    // the AO tier's cheaper entry point is then the whole cost.
    std::array<double, 9> second{};
};

// One atom index per AO slot, from the envelopes' own ranges: an AO's motion
// reaches the atom it sits on and no other, so this is the map the orbital-motion
// term is placed by. The map is also the check that the envelope set covers the
// basis exactly once - a set that leaves a slot unclaimed would place that AO's
// motion on no atom at all, silently.
qcx::Result<std::vector<std::size_t>> BuildAtomOfAo(std::span<const ShellEnvelope> envelopes,
                                                    std::size_t aoCount) {
    std::vector<std::size_t> atomOfAo(aoCount, kNoAtom);

    for (const ShellEnvelope& shell : envelopes)
    {
        if (shell.aoOffset + shell.functionCount > aoCount)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "EvaluateXcGradient: a shell envelope reaches past AOCount()"});
        }

        for (std::size_t ao = 0; ao < shell.functionCount; ++ao)
        {
            std::size_t& slot = atomOfAo[shell.aoOffset + ao];

            if (slot != kNoAtom)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument,
                               "EvaluateXcGradient: two shell envelopes claim the same AO slot"});
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
                           "EvaluateXcGradient: the shell envelopes do not cover every AO"});
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
        // same slot twice, and accumulating them would double every diagonal
        // entry of every AO.
        expanded[3 * row + column] = entry;
        expanded[3 * column + row] = entry;
    }

    return expanded;
}

// One point's fetched AO slice: the values and their first and second
// derivatives as the evaluator wrote them, and the AO slots the envelopes kept.
// Carried as one object because the three arrays are the same tier read at three
// strides, and a caller that handed two of them over in the other order would
// read one array's tail as another's data.
struct AoSlice {
    std::span<const double> values;
    std::span<const double> gradients;
    std::span<const double> hessians;
    std::span<const std::size_t> selectedAos;
};

// The two contractions the density's chain rule produces at a point: (D phi)_mu
// and (D grad phi)_mu. Both are scratch the walk reuses across points and both
// are indexed by AO slot, so they travel together rather than as two vectors of
// the same type and different lengths.
struct PointContractions {
    std::vector<double> value;
    std::vector<double> gradient;
};

// One spin's density field at a point, from the point's fetched AO slice.
PointField BuildPointField(const Eigen::MatrixXd& density,
                           const AoSlice& aos,
                           PointContractions& contractions,
                           bool withSecond) {
    const PointDensity contracted = internal::ContractSelectedSpinDensity(density,
                                                                          aos.values,
                                                                          aos.gradients,
                                                                          aos.selectedAos,
                                                                          contractions.value,
                                                                          contractions.gradient);

    PointField field;
    field.rho = contracted.rho;
    field.gradient = contracted.gradient;

    if (!withSecond)
    {
        return field;
    }

    // The field's second derivative,
    //     hess rho = 2 sum_mu [ (D phi)_mu hess phi_mu
    //                           + grad phi_mu (D grad phi)_mu^T ],
    // from the two contractions the density already needed and the AO tier's
    // second derivative. The outer product is not symmetric on its own, so both
    // of its halves are written and the result is symmetric by construction.
    for (const std::size_t mu : aos.selectedAos)
    {
        const double value = 2.0 * contractions.value[mu];
        const std::array<double, 3> gradientMu = {
            aos.gradients[3 * mu + 0], aos.gradients[3 * mu + 1], aos.gradients[3 * mu + 2]};
        const std::array<double, 3> contractedGradient = {contractions.gradient[3 * mu + 0],
                                                          contractions.gradient[3 * mu + 1],
                                                          contractions.gradient[3 * mu + 2]};
        const std::array<double, 9> curvature = ExpandHessian(aos.hessians, mu);

        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                field.second[3 * row + column] +=
                    value * curvature[3 * row + column] +
                    2.0 * gradientMu[row] * contractedGradient[column];
            }
        }
    }

    return field;
}

// H x, for the symmetric 3x3 the field carries in row-major order.
std::array<double, 3> SymmetricTimes(const std::array<double, 9>& matrix,
                                     const std::array<double, 3>& vector) {
    std::array<double, 3> out{};

    for (std::size_t row = 0; row < 3; ++row)
    {
        for (std::size_t column = 0; column < 3; ++column)
        {
            out[row] += matrix[3 * row + column] * vector[column];
        }
    }

    return out;
}

// The spatial gradient of the integrand at a point, by the chain rule through
// the fields:
//     grad e = sum_s vrho_s grad rho_s
//            + sum_st vsigma_st grad sigma_st,
//     grad sigma_st = (hess rho_s) grad rho_t + (hess rho_t) grad rho_s.
// This is the term that makes the grid's motion visible: e is a function of the
// point, so a point that moves with its atom carries the integrand with it.
std::array<double, 3> GradientOfEnergyDensity(const excgrid::XcKernelValue& kernel,
                                              const PointField& alpha,
                                              const PointField& beta,
                                              bool withSecond) {
    std::array<double, 3> result{};

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        result[axis] = kernel.vrhoA * alpha.gradient[axis] + kernel.vrhoB * beta.gradient[axis];
    }

    if (!withSecond)
    {
        return result;
    }

    // The diagonal pairs are counted twice: sigma_aa and sigma_bb each take both
    // of their derivatives from the same field, so their curvature is doubled.
    const std::array<double, 3> sigmaAa = SymmetricTimes(alpha.second, alpha.gradient);
    const std::array<double, 3> sigmaAb = SymmetricTimes(alpha.second, beta.gradient);
    const std::array<double, 3> sigmaBa = SymmetricTimes(beta.second, alpha.gradient);
    const std::array<double, 3> sigmaBb = SymmetricTimes(beta.second, beta.gradient);

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        result[axis] += kernel.vsigmaAa * 2.0 * sigmaAa[axis] +
                        kernel.vsigmaAb * (sigmaAb[axis] + sigmaBa[axis]) +
                        kernel.vsigmaBb * 2.0 * sigmaBb[axis];
    }

    return result;
}

// One spin's two density-derivative sums at a point, one entry per atom:
//     P^s_A = sum_{mu in A} (D_s phi)_mu grad phi_mu
//     Q^s_A = sum_{mu in A} [ (D_s grad phi)_mu (x) grad phi_mu
//                             + (D_s phi)_mu hess phi_mu ]
// evaluated over the KEPT slots, because a slot the selection dropped is a term
// the energy dropped too. These two sums are the whole of the density's
// dependence on the atomic orbitals' motion, so the functional's chain rule
// reads them once for rho and once for each sigma pair.
struct SpinSums {
    std::vector<std::array<double, 3>> position;
    std::vector<std::array<double, 9>> curvature;

    explicit SpinSums(std::size_t atomCount) :
        position(atomCount, std::array<double, 3>{}),
        curvature(atomCount, std::array<double, 9>{}) {}

    void Clear() noexcept {
        std::fill(position.begin(), position.end(), std::array<double, 3>{});
        std::fill(curvature.begin(), curvature.end(), std::array<double, 9>{});
    }
};

// Fills one spin's sums from the point's fetched slice and the contractions its
// density field was built from.
//
// The second sum is indexed [nuclear coordinate][gradient direction]. Its two
// terms are not interchangeable: one pairs the AO gradient's nuclear-coordinate
// component with the contracted gradient's field direction, the other is the
// AO's own curvature, and only the second is symmetric.
void AccumulateSpinSums(SpinSums& sums,
                        const AoSlice& aos,
                        const PointContractions& contractions,
                        std::span<const std::size_t> atomOfAo,
                        bool withSecond,
                        std::size_t& terms) {
    for (const std::size_t mu : aos.selectedAos)
    {
        const std::size_t atom = atomOfAo[mu];
        const double value = contractions.value[mu];
        const std::array<double, 3> contracted = {contractions.gradient[3 * mu + 0],
                                                  contractions.gradient[3 * mu + 1],
                                                  contractions.gradient[3 * mu + 2]};
        const std::array<double, 3> gradientMu = {
            aos.gradients[3 * mu + 0], aos.gradients[3 * mu + 1], aos.gradients[3 * mu + 2]};
        const std::array<double, 9> curvature =
            withSecond ? ExpandHessian(aos.hessians, mu) : std::array<double, 9>{};

        for (std::size_t row = 0; row < 3; ++row)
        {
            sums.position[atom][row] += value * gradientMu[row];

            // The curvature sum exists only for the sigma pairs, so a functional
            // that reads no density gradient pays neither the AO tier's second
            // derivative nor the accumulation of it.
            for (std::size_t column = 0; withSecond && column < 3; ++column)
            {
                sums.curvature[atom][3 * row + column] +=
                    gradientMu[row] * contracted[column] + value * curvature[3 * row + column];
            }
        }

        ++terms;
    }
}

// Q v, for the row-major 3x3 Q the sums carry, and Q^T v.
std::array<double, 3> MatrixTimes(const std::array<double, 9>& matrix,
                                  const std::array<double, 3>& vector) {
    std::array<double, 3> out{};

    for (std::size_t row = 0; row < 3; ++row)
    {
        for (std::size_t column = 0; column < 3; ++column)
        {
            out[row] += matrix[3 * row + column] * vector[column];
        }
    }

    return out;
}

// The orbitals' motion at a point, from the two sums the density's chain rule
// produces:
//     d rho_s / dR_A          = -2 P^s_A
//     d grad rho_s / dR_A     = -2 Q^s_A
//     d sigma_st / dR_A       = -2 [ (Q^s_A grad rho_t) + (Q^t_A grad rho_s) ]
// and the functional's own chain rule weights them by its partials. Both signs
// are the same one: the AO moves against the atom's displacement, which is why
// the term is subtracted rather than added. The two sigma terms are the same
// matrix-vector product twice, so neither is transposed - a transpose here is
// invisible whenever the pair is diagonal and wrong on the cross pair.
void AccumulateOrbitalMotion(Eigen::VectorXd& gradient,
                             const SpinSums& alphaSums,
                             const SpinSums& betaSums,
                             const PointField& alpha,
                             const PointField& beta,
                             const excgrid::XcKernelValue& kernel,
                             double weight) {
    for (std::size_t atom = 0; atom < alphaSums.position.size(); ++atom)
    {
        const std::array<double, 3>& positionAlpha = alphaSums.position[atom];
        const std::array<double, 3>& positionBeta = betaSums.position[atom];
        const std::array<double, 9>& curvatureAlpha = alphaSums.curvature[atom];
        const std::array<double, 9>& curvatureBeta = betaSums.curvature[atom];
        const std::array<double, 3> sigmaAa = MatrixTimes(curvatureAlpha, alpha.gradient);
        const std::array<double, 3> sigmaBb = MatrixTimes(curvatureBeta, beta.gradient);
        const std::array<double, 3> sigmaAb = MatrixTimes(curvatureAlpha, beta.gradient);
        const std::array<double, 3> sigmaBa = MatrixTimes(curvatureBeta, alpha.gradient);

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            // The diagonal pairs contribute their two terms twice, because the
            // chain rule takes both of a pair's derivatives from the same field.
            const double value = kernel.vrhoA * positionAlpha[axis] +
                                 kernel.vrhoB * positionBeta[axis] +
                                 2.0 * kernel.vsigmaAa * sigmaAa[axis] +
                                 kernel.vsigmaAb * (sigmaAb[axis] + sigmaBa[axis]) +
                                 2.0 * kernel.vsigmaBb * sigmaBb[axis];
            const Eigen::Index component = static_cast<Eigen::Index>(3 * atom + axis);
            gradient[component] -= 2.0 * weight * value;
        }
    }
}

} // namespace

qcx::Result<XcGradientEvaluation> EvaluateXcGradient(
    std::span<const excgrid::Block> blocks,
    const excgrid::GridDerivativeProvider& provider,
    const AoEvaluator& evaluator,
    const excgrid::XcFunctional& functional,
    std::span<const ShellEnvelope> envelopes,
    const XcIntegrationThresholds& thresholds,
    const XcBatchSettings& batch,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta) {
    const Eigen::Index n = static_cast<Eigen::Index>(evaluator.AOCount());

    if (densityAlpha.rows() != n || densityAlpha.cols() != n || densityBeta.rows() != n ||
        densityBeta.cols() != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "EvaluateXcGradient: both densities must be AOCount() x AOCount()"});
    }

    auto atomOfAo = BuildAtomOfAo(envelopes, evaluator.AOCount());

    if (!atomOfAo.has_value())
    {
        return std::unexpected(atomOfAo.error());
    }

    // The molecule's atom count, read off the envelopes rather than assumed: the
    // nuclear coordinates the provider's arrays are indexed by are 3 per atom,
    // and an envelope naming an atom past the end would place a contribution
    // outside the gradient.
    std::size_t atomCount = 0;

    for (const ShellEnvelope& shell : envelopes)
    {
        atomCount = std::max(atomCount, shell.atomIndex + 1);
    }

    if (atomCount == 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "EvaluateXcGradient: the shell envelopes name no atom"});
    }

    std::size_t largestBlock = 0;

    for (const excgrid::Block& block : blocks)
    {
        largestBlock = std::max(largestBlock, block.pointCount);
    }

    const std::size_t nuclearCoordinateCount = 3U * atomCount;

    XcGradientEvaluation evaluation;
    evaluation.thresholds = thresholds;
    evaluation.gradient = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nuclearCoordinateCount));

    const bool usesGradient = functional.UsesGradient();

    // The per-point scratch, reused across the whole grid: the two
    // contractions a point's density field is built from, and the two spins'
    // orbital-motion sums, which are one entry per atom.
    PointContractions contractions;
    contractions.value.resize(evaluator.AOCount());
    contractions.gradient.resize(3 * evaluator.AOCount());
    SpinSums alphaSums(atomCount);
    SpinSums betaSums(atomCount);

    // The batch buffers, sized once for the whole walk rather than per block:
    // one slice of AO values, gradients and - for a functional that consumes the
    // density gradients - second derivatives per batched point, plus that point's
    // selection in the two flat lists the fetch and the assembly both walk. The
    // size is clamped to the largest block, so a batch can never span blocks.
    const std::size_t bufferPoints =
        batch.pointBatch == 0 ? largestBlock : std::min(batch.pointBatch, largestBlock);
    const std::size_t aoCount = evaluator.AOCount();

    std::vector<double> batchValues(bufferPoints * aoCount);
    std::vector<double> batchGradients(bufferPoints * 3 * aoCount);
    std::vector<double> batchHessians(usesGradient ? bufferPoints * 6 * aoCount : 0);
    std::vector<std::size_t> batchShells(bufferPoints * envelopes.size());
    std::vector<std::size_t> batchAos(bufferPoints * aoCount);
    std::vector<std::size_t> batchShellOffsets(bufferPoints + 1);
    std::vector<std::size_t> batchAoOffsets(bufferPoints + 1);
    std::vector<double> batchWeights(bufferPoints);

    // The provider's buffers, likewise sized once for the largest block. The
    // request is the one the contract documents: both parts at the first order,
    // indexed by the caller's own 3N nuclear coordinates.
    excgrid::DerivativeRequest request;
    request.parts = BothParts();
    request.order = excgrid::DerivativeOrder::kFirst;
    request.nuclearCoordinateCount = nuclearCoordinateCount;

    const excgrid::BlockDerivativeSizes providerSizes = excgrid::SizeOf(request, largestBlock);
    std::vector<double> positionFirst(providerSizes.positionFirst);
    std::vector<double> weightFirst(providerSizes.weightFirst);

    // The screening weights cost one pass over each spin's density matrix,
    // O(AOCount()^2), against the O(points x AOCount()^2) of the assembly.
    const std::vector<double> flatAlpha = internal::FlattenRowMajor(densityAlpha);
    auto weightsAlpha = ShellDensityWeights(envelopes, flatAlpha, aoCount);

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
        auto computed = ShellDensityWeights(envelopes, flatBeta, aoCount);

        if (!computed.has_value())
        {
            return std::unexpected(computed.error());
        }

        weightsBeta = std::move(*computed);
    }

    evaluation.counts.densityWeightTerms = aoCount * aoCount * (sharedSpinDensity ? 1 : 2);

    for (const excgrid::Block& block : blocks)
    {
        if (block.points.size() != block.pointCount || block.weights.size() != block.pointCount ||
            block.atomIndex.size() != block.pointCount)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "EvaluateXcGradient: a block's points, weights and owning atoms must "
                           "each hold its point count entries"});
        }

        const excgrid::BlockDerivativeSizes sizes = excgrid::SizeOf(request, block.pointCount);
        excgrid::BlockDerivatives derivatives;
        derivatives.positionFirst = std::span<double>(positionFirst.data(), sizes.positionFirst);
        derivatives.weightFirst = std::span<double>(weightFirst.data(), sizes.weightFirst);
        derivatives.atomIndex = block.atomIndex;

        const excgrid::DerivativeStatus status = provider.Evaluate(block, request, derivatives);

        if (status != excgrid::DerivativeStatus::kOk)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "EvaluateXcGradient: the grid derivative provider refused a block: " +
                               std::string(DerivativeStatusName(status))});
        }

        evaluation.counts.weightDerivativeTerms += sizes.weightFirst;
        evaluation.counts.coordinateDerivativeTerms += sizes.positionFirst;

        const std::size_t batchSize =
            batch.pointBatch == 0 ? block.pointCount : std::min(batch.pointBatch, block.pointCount);

        for (std::size_t start = 0; start < block.pointCount; start += batchSize)
        {
            const std::size_t count = std::min(batchSize, block.pointCount - start);
            ++evaluation.counts.batches;

            // Phase one: select and FETCH, exactly the selection the energy path
            // makes at the same point under the same tolerance.
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

                for (std::size_t shell = 0; shell < envelopes.size(); ++shell)
                {
                    ++evaluation.counts.shellTests;

                    const double spinWeight = std::max((*weightsAlpha)[shell], weightsBeta[shell]);

                    if (!ShellIsSignificant(envelopes[shell],
                                            block.points[point],
                                            spinWeight,
                                            thresholds.screeningTolerance))
                    {
                        continue;
                    }

                    ++evaluation.counts.shellKeeps;
                    batchShells[shellCursor++] = shell;

                    for (std::size_t ao = 0; ao < envelopes[shell].functionCount; ++ao)
                    {
                        batchAos[aoCursor++] = envelopes[shell].aoOffset + ao;
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

                const std::span<double> values(batchValues.data() + index * aoCount, aoCount);
                const std::span<double> gradients(batchGradients.data() + index * 3 * aoCount,
                                                  3 * aoCount);
                const std::span<const std::size_t> selection(
                    batchShells.data() + batchShellOffsets[index],
                    shellCursor - batchShellOffsets[index]);

                if (usesGradient)
                {
                    const std::span<double> hessians(batchHessians.data() + index * 6 * aoCount,
                                                     6 * aoCount);
                    evaluator.EvaluateDerivativesSelected(
                        block.points[point], selection, values, gradients, hessians);
                } else
                {
                    // The energy path's LDA tier is the values alone, and that is
                    // not enough here: the point's own motion and the orbitals'
                    // motion are derivatives of the density, so this walk reads the
                    // first derivatives even from a functional that never asks for
                    // them.
                    evaluator.EvaluateGradientsSelected(
                        block.points[point], selection, values, gradients);
                }
            }

            batchShellOffsets[count] = shellCursor;
            batchAoOffsets[count] = aoCursor;

            // Phase two: assemble every term of the gradient from the slice phase
            // one filled, and nothing else.
            for (std::size_t index = 0; index < count; ++index)
            {
                if (batchShellOffsets[index] == batchShellOffsets[index + 1])
                {
                    continue;
                }

                const std::size_t point = start + index;
                const std::span<const double> values(batchValues.data() + index * aoCount, aoCount);
                const std::span<const double> gradients(batchGradients.data() + index * 3 * aoCount,
                                                        3 * aoCount);
                const std::span<const double> hessians(batchHessians.data() + index * 6 * aoCount,
                                                       6 * aoCount);
                const std::span<const std::size_t> selectedAos(
                    batchAos.data() + batchAoOffsets[index],
                    batchAoOffsets[index + 1] - batchAoOffsets[index]);
                const AoSlice aos{values, gradients, hessians, selectedAos};

                const PointField fieldA =
                    BuildPointField(densityAlpha, aos, contractions, usesGradient);

                // The alpha sums read the contractions the alpha field was just
                // built from, so they are taken before the beta field overwrites
                // them. Nothing else about the point is spin-ordered.
                alphaSums.Clear();
                AccumulateSpinSums(alphaSums,
                                   aos,
                                   contractions,
                                   *atomOfAo,
                                   usesGradient,
                                   evaluation.counts.aoDerivativeTerms);

                const PointField fieldB =
                    BuildPointField(densityBeta, aos, contractions, usesGradient);

                betaSums.Clear();
                AccumulateSpinSums(betaSums,
                                   aos,
                                   contractions,
                                   *atomOfAo,
                                   usesGradient,
                                   evaluation.counts.aoDerivativeTerms);

                const Eigen::Vector3d gradientA(
                    fieldA.gradient[0], fieldA.gradient[1], fieldA.gradient[2]);
                const Eigen::Vector3d gradientB(
                    fieldB.gradient[0], fieldB.gradient[1], fieldB.gradient[2]);

                const excgrid::XcKernelValue kernel = functional.Evaluate(fieldA.rho,
                                                                          fieldB.rho,
                                                                          gradientA.dot(gradientA),
                                                                          gradientA.dot(gradientB),
                                                                          gradientB.dot(gradientB));

                const double weight = batchWeights[index];
                evaluation.energy += weight * kernel.exc;

                // The weight's motion: the quadrature weight is a function of the
                // geometry, and the integrand rides it.
                const double* weightRow = weightFirst.data() + point * nuclearCoordinateCount;

                for (std::size_t coordinate = 0; coordinate < nuclearCoordinateCount; ++coordinate)
                {
                    evaluation.gradient[static_cast<Eigen::Index>(coordinate)] +=
                        kernel.exc * weightRow[coordinate];
                }

                // The point's motion: the integrand is a function of position, so
                // a point that moves with its atom carries its own value with it.
                const std::array<double, 3> energyDensityGradient =
                    GradientOfEnergyDensity(kernel, fieldA, fieldB, usesGradient);
                const double* positionRow =
                    positionFirst.data() + point * 3 * nuclearCoordinateCount;

                for (std::size_t direction = 0; direction < 3; ++direction)
                {
                    const double value = weight * energyDensityGradient[direction];
                    const double* row = positionRow + direction * nuclearCoordinateCount;

                    for (std::size_t coordinate = 0; coordinate < nuclearCoordinateCount;
                         ++coordinate)
                    {
                        evaluation.gradient[static_cast<Eigen::Index>(coordinate)] +=
                            value * row[coordinate];
                    }
                }

                // The orbitals' motion, from the two sums per spin.
                AccumulateOrbitalMotion(
                    evaluation.gradient, alphaSums, betaSums, fieldA, fieldB, kernel, weight);
            }
        }
    }

    evaluation.counts.aoSlotsDense = evaluation.counts.points * aoCount;
    evaluation.counts.contractionPairsDense = 2 * evaluation.counts.points * aoCount * aoCount;
    evaluation.counts.aoDerivativeTermsDense = 2 * evaluation.counts.points * aoCount;

    return evaluation;
}

qcx::Result<XcGradientEvaluation> EvaluateXcGradient(
    const XcGridEngine& engine,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    const excgrid::GridDerivativeProvider& provider,
    const XcIntegrationThresholds& thresholds,
    const XcBatchSettings& batch) {
    return EvaluateXcGradient(engine.Grid().Blocks(),
                              provider,
                              engine.Evaluator(),
                              engine.Functional(),
                              engine.ShellEnvelopes(),
                              thresholds,
                              batch,
                              densityAlpha,
                              densityBeta);
}

} // namespace qcx::grid
