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

// One spin's density field at a point: the value, its first derivative, the
// kinetic-energy density and - for a functional that consumes the density
// gradients - the second derivative.
struct PointField {
    double rho = 0.0;
    /// 1/2 sum_mu nu D_mu nu grad phi_mu . grad phi_nu, the same contraction
    /// the energy path runs, so the two paths' kernels see one number.
    double tau = 0.0;
    std::array<double, 3> gradient{};
    /// grad tau, written only for a functional that reads tau: the same
    /// contraction against the AO tier's second derivative.
    std::array<double, 3> tauGradient{};
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
//
// The refusal messages name the operation that asked, so a refusal from either
// walk is attributable to the call the caller made rather than to the other one.
qcx::Result<std::vector<std::size_t>> BuildAtomOfAo(std::string_view operation,
                                                    std::span<const ShellEnvelope> envelopes,
                                                    std::size_t aoCount) {
    std::vector<std::size_t> atomOfAo(aoCount, kNoAtom);

    for (const ShellEnvelope& shell : envelopes)
    {
        if (shell.aoOffset + shell.functionCount > aoCount)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           std::string(operation) + ": a shell envelope reaches past AOCount()"});
        }

        for (std::size_t ao = 0; ao < shell.functionCount; ++ao)
        {
            std::size_t& slot = atomOfAo[shell.aoOffset + ao];

            if (slot != kNoAtom)
            {
                return std::unexpected(qcx::Error{
                    qcx::ErrorCode::kInvalidArgument,
                    std::string(operation) + ": two shell envelopes claim the same AO slot"});
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
                           std::string(operation) + ": the shell envelopes do not cover every AO"});
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
// The two flags select independent pieces of the same second-derivative tier:
// the density's curvature belongs to the sigma pairs, tau's motion to vtau, and
// a functional that reads neither pays for neither.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
PointField BuildPointField(const Eigen::MatrixXd& density,
                           const AoSlice& aos,
                           PointContractions& contractions,
                           bool withSecond,
                           bool withTau) {
    const PointDensity contracted = internal::ContractSelectedSpinDensity(density,
                                                                          aos.values,
                                                                          aos.gradients,
                                                                          aos.selectedAos,
                                                                          contractions.value,
                                                                          contractions.gradient);

    PointField field;
    field.rho = contracted.rho;
    field.tau = contracted.tau;
    field.gradient = contracted.gradient;

    if (!withSecond && !withTau)
    {
        return field;
    }

    for (const std::size_t mu : aos.selectedAos)
    {
        const std::array<double, 3> contractedGradient = {contractions.gradient[3 * mu + 0],
                                                          contractions.gradient[3 * mu + 1],
                                                          contractions.gradient[3 * mu + 2]};
        const std::array<double, 9> curvature = ExpandHessian(aos.hessians, mu);

        if (withSecond)
        {
            // The field's second derivative,
            //     hess rho = 2 sum_mu [ (D phi)_mu hess phi_mu
            //                           + grad phi_mu (D grad phi)_mu^T ],
            // from the two contractions the density already needed and the AO
            // tier's second derivative. The outer product is not symmetric on
            // its own, so both of its halves are written and the result is
            // symmetric by construction.
            const double value = 2.0 * contractions.value[mu];
            const std::array<double, 3> gradientMu = {
                aos.gradients[3 * mu + 0], aos.gradients[3 * mu + 1], aos.gradients[3 * mu + 2]};

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

        if (withTau)
        {
            //     d_k tau = sum_mu sum_j (D grad phi)_mu^j hess phi_mu[j][k],
            // with no surviving 1/2: it belongs to tau's definition and to the
            // potential, not to the field's own derivative. The row runs over
            // the AO gradient's components, the column over the derivative
            // direction.
            for (std::size_t row = 0; row < 3; ++row)
            {
                for (std::size_t column = 0; column < 3; ++column)
                {
                    field.tauGradient[column] +=
                        contractedGradient[row] * curvature[3 * row + column];
                }
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

// One point's kernel call, by named component rather than by position: the
// positional adapter cannot carry tau and reports a refusal as zeros. The seven
// components the walk supplies are all set, so the input mask is the collinear
// set; the result's named first derivatives are read back, and a slot the
// functional did not produce stays at its initial zero. A refusal is returned
// as an Error naming the status, never as a silent zero energy.
qcx::Result<excgrid::XcKernelValue> EvaluateKernel(
    std::string_view operation,
    const excgrid::XcFunctional& functional,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const PointField& alpha,
    const PointField& beta) {
    const Eigen::Vector3d gradientA(alpha.gradient[0], alpha.gradient[1], alpha.gradient[2]);
    const Eigen::Vector3d gradientB(beta.gradient[0], beta.gradient[1], beta.gradient[2]);

    excgrid::PointInputs inputs;
    inputs.Set(excgrid::Component::RhoA, alpha.rho);
    inputs.Set(excgrid::Component::RhoB, beta.rho);
    inputs.Set(excgrid::Component::SigmaAa, gradientA.dot(gradientA));
    inputs.Set(excgrid::Component::SigmaAb, gradientA.dot(gradientB));
    inputs.Set(excgrid::Component::SigmaBb, gradientB.dot(gradientB));
    inputs.Set(excgrid::Component::TauA, alpha.tau);
    inputs.Set(excgrid::Component::TauB, beta.tau);

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

// The spatial gradient of the integrand at a point, by the chain rule through
// the fields:
//     grad e = sum_s vrho_s grad rho_s
//            + sum_s vtau_s grad tau_s
//            + sum_st vsigma_st grad sigma_st,
//     grad sigma_st = (hess rho_s) grad rho_t + (hess rho_t) grad rho_s.
// This is the term that makes the grid's motion visible: e is a function of the
// point, so a point that moves with its atom carries the integrand with it.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::array<double, 3> GradientOfEnergyDensity(const excgrid::XcKernelValue& kernel,
                                              const PointField& alpha,
                                              const PointField& beta,
                                              bool withSecond,
                                              bool withTau) {
    std::array<double, 3> result{};

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        result[axis] = kernel.vrhoA * alpha.gradient[axis] + kernel.vrhoB * beta.gradient[axis];

        if (withTau)
        {
            result[axis] +=
                kernel.vtauA * alpha.tauGradient[axis] + kernel.vtauB * beta.tauGradient[axis];
        }
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

// One spin's density-derivative sums at a point, one entry per atom:
//     P^s_A = sum_{mu in A} (D_s phi)_mu grad phi_mu
//     Q^s_A = sum_{mu in A} [ (D_s grad phi)_mu (x) grad phi_mu
//                             + (D_s phi)_mu hess phi_mu ]
//     S^s_A = sum_{mu in A} sum_j (D_s grad phi)_mu^j hess phi_mu[j][.]
// evaluated over the KEPT slots, because a slot the selection dropped is a term
// the energy dropped too. These are the whole of the density's and the
// kinetic-energy density's dependence on the atomic orbitals' motion, so the
// functional's chain rule reads them once for rho, once for each sigma pair,
// and once for tau.
struct SpinSums {
    std::vector<std::array<double, 3>> position;
    std::vector<std::array<double, 9>> curvature;
    std::vector<std::array<double, 3>> tau;

    explicit SpinSums(std::size_t atomCount) :
        position(atomCount, std::array<double, 3>{}), curvature(atomCount, std::array<double, 9>{}),
        tau(atomCount, std::array<double, 3>{}) {}

    void Clear() noexcept {
        std::fill(position.begin(), position.end(), std::array<double, 3>{});
        std::fill(curvature.begin(), curvature.end(), std::array<double, 9>{});
        std::fill(tau.begin(), tau.end(), std::array<double, 3>{});
    }
};

// Fills one spin's sums from the point's fetched slice and the contractions its
// density field was built from.
//
// The second sum is indexed [nuclear coordinate][gradient direction]. Its two
// terms are not interchangeable: one pairs the AO gradient's nuclear-coordinate
// component with the contracted gradient's field direction, the other is the
// AO's own curvature, and only the second is symmetric. The tau sum needs only
// the second term, read as the contracted gradient along the AO's own row.
// The two flags select the two sums independently, so a functional that reads
// tau and not sigma pays for the tau sum alone.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void AccumulateSpinSums(SpinSums& sums,
                        const AoSlice& aos,
                        const PointContractions& contractions,
                        std::span<const std::size_t> atomOfAo,
                        bool withSecond,
                        bool withTau,
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
            (withSecond || withTau) ? ExpandHessian(aos.hessians, mu) : std::array<double, 9>{};

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

        if (withTau)
        {
            // S^s_A[k] = sum_j (D grad phi)_mu^j hess phi_mu[j][k]: the row is
            // the AO gradient's component, the column the derivative direction.
            for (std::size_t row = 0; row < 3; ++row)
            {
                for (std::size_t column = 0; column < 3; ++column)
                {
                    sums.tau[atom][column] += contracted[row] * curvature[3 * row + column];
                }
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

// The orbitals' motion at a point, from the sums the density's and the
// kinetic-energy density's chain rules produce:
//     d rho_s / dR_A          = -2 P^s_A
//     d grad rho_s / dR_A     = -2 Q^s_A
//     d sigma_st / dR_A       = -2 [ (Q^s_A grad rho_t) + (Q^t_A grad rho_s) ]
//     d tau_s / dR_A          = -S^s_A
// and the functional's own chain rule weights them by its partials. Both signs
// are the same one: the AO moves against the atom's displacement, which is why
// the term is subtracted rather than added. The tau term is the odd one out by
// a factor of two: the 1/2 in tau's definition cancels the two density index
// slots, so it carries coefficient 1 beside the same minus, while the rho and
// sigma terms carry 2 from the two slots of their contraction. Writing it at 2
// is a factor-2 defect no internally consistent check sees. The two sigma terms
// are the same matrix-vector product twice, so neither is transposed - a
// transpose here is invisible whenever the pair is diagonal and wrong on the
// cross pair.
void AccumulateOrbitalMotion(std::span<double> gradient,
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
        const std::array<double, 3>& tauAlpha = alphaSums.tau[atom];
        const std::array<double, 3>& tauBeta = betaSums.tau[atom];
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
            const double tauValue = kernel.vtauA * tauAlpha[axis] + kernel.vtauB * tauBeta[axis];
            gradient[3 * atom + axis] -= 2.0 * weight * value + weight * tauValue;
        }
    }
}

// The components whose second total response this walk carries: the two spin
// densities. The sigma invariants and the kinetic-energy densities need one
// more x derivative of every AO through their own second response - the
// third-derivative tier the evaluator does not supply - so a functional that
// reads one of them is refused rather than answered with the density channel.
bool IsDensityComponent(excgrid::Component id) noexcept {
    return id == excgrid::Component::RhoA || id == excgrid::Component::RhoB;
}

// The spin a density component names: 0 alpha, 1 beta. Only the two identifiers
// the density channel accepts reach it.
std::size_t SpinOfDensity(excgrid::Component id) noexcept {
    return id == excgrid::Component::RhoA ? 0U : 1U;
}

// A mask's component names, in identifier order, so a refusal names what it
// could not answer for rather than saying that something is missing.
std::string MaskComponentNames(const excgrid::ComponentMask& mask) {
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

        names += std::string(excgrid::Describe(id).name);
    }

    return names;
}

// The refusal the electronic channel owes before the walk starts, or an empty
// string when it can answer.
//
// A functional that reads nothing has no second response to take and is
// answered by the weight terms alone. One that reads past the two densities is
// refused by name: its sigma or tau second response differentiates the AO
// Hessian once more, and answering it with the density channel alone would be a
// plausible wrong number rather than a refusal. One that reads only the
// densities but declares no second-derivative tier is refused too, because the
// tier is how a functional says which components it can be differentiated twice
// in.
std::string ChannelRefusal(const excgrid::XcFunctional& functional) {
    const excgrid::ComponentMask required = functional.RequiredMask();

    if (required.bits == 0)
    {
        return {};
    }

    excgrid::ComponentMask outside;

    for (std::size_t index = 0; index < excgrid::kComponentCapacity; ++index)
    {
        const auto id = static_cast<excgrid::Component>(index);

        if (required.Test(id) && !IsDensityComponent(id))
        {
            outside.Set(id);
        }
    }

    if (outside.bits != 0)
    {
        return "the density channel does not span " + MaskComponentNames(outside) +
               ": their second response needs the third-derivative AO tier";
    }

    const excgrid::KernelStatus status =
        excgrid::SecondDerivativeStatus(required, required, functional.SecondDerivativeMask());

    if (status == excgrid::KernelStatus::kRefusedUnsupportedCapability)
    {
        return "the functional declares no second-derivative tier";
    }

    if (status == excgrid::KernelStatus::kRefusedSecondDerivativeCoverage)
    {
        return "the functional's second-derivative tier does not span " +
               MaskComponentNames(excgrid::UnspannedRequestComponents(
                   required, required, functional.SecondDerivativeMask()));
    }

    return {};
}

// The density's second-order orbital partials at a point, per atom:
//     U_A[a][b] = sum_{mu in A} (D phi)_mu hess phi_mu[a][b]
//     R_A[b][mu] = sum_{nu in A} D_mu nu d_b phi_nu .
// U is the curvature's density half. R is the blocked gradient contraction: the
// density's second response pairs two atoms, pairing needs the contraction's
// two slots on different atoms, and the block keeps B's contribution to every
// mu rather than summing it away.
//
// Only the kept atoms and the kept slots are ever read, so only they are
// cleared.
struct SecondOrderSums {
    std::vector<std::array<double, 9>> curvature;
    std::vector<double> blocked;
    std::size_t aoCount = 0;

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    SecondOrderSums(std::size_t atomCount, std::size_t aos) :
        curvature(atomCount, std::array<double, 9>{}), blocked(atomCount * 3 * aos, 0.0),
        aoCount(aos) {}

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] double& Blocked(std::size_t atom,
                                  std::size_t direction,
                                  std::size_t ao) noexcept {
        return blocked[(atom * 3 + direction) * aoCount + ao];
    }

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] double Blocked(std::size_t atom,
                                 std::size_t direction,
                                 std::size_t ao) const noexcept {
        return blocked[(atom * 3 + direction) * aoCount + ao];
    }
};

void ClearSecondOrderSums(SecondOrderSums& sums,
                          const AoSlice& aos,
                          std::span<const std::size_t> keptAtoms) noexcept {
    for (const std::size_t atom : keptAtoms)
    {
        sums.curvature[atom] = std::array<double, 9>{};

        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            for (const std::size_t ao : aos.selectedAos)
            {
                sums.Blocked(atom, direction, ao) = 0.0;
            }
        }
    }
}

// Fills one spin's second-order partials from the point's slice and the
// contractions its density field was built from. The contracted gradient is
// blocked by the atom its second slot sits on, so the pair term reads one
// block per atom instead of a point-sized pair array.
void AccumulateSecondOrderSums(SecondOrderSums& sums,
                               const AoSlice& aos,
                               const PointContractions& contractions,
                               const Eigen::MatrixXd& density,
                               std::span<const std::size_t> atomOfAo) {
    for (const std::size_t mu : aos.selectedAos)
    {
        const std::size_t atom = atomOfAo[mu];
        const double value = contractions.value[mu];
        const std::array<double, 9> curvature = ExpandHessian(aos.hessians, mu);

        for (std::size_t entry = 0; entry < 9; ++entry)
        {
            sums.curvature[atom][entry] += value * curvature[entry];
        }

        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            for (const std::size_t nu : aos.selectedAos)
            {
                sums.Blocked(atomOfAo[nu], direction, mu) +=
                    density(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) *
                    aos.gradients[3 * nu + direction];
            }
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

    auto atomOfAo = BuildAtomOfAo("EvaluateXcGradient", envelopes, evaluator.AOCount());

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

    // What the functional reads decides what the walk fetches. Tau needs the AO
    // gradients and their second derivatives even when sigma is not read, so the
    // AO TIER's gate is the functional's own gradient flag extended by the two
    // tau bits; a tau-only functional otherwise gets a zero-size Hessian buffer.
    // The density's own curvature stays the sigma pairs' business, so the FIELD
    // gate is UsesGradient() alone: vtau does not weight a second derivative of
    // rho, only the contracted gradient's motion.
    const excgrid::ComponentMask required = functional.RequiredMask();
    const bool usesTau =
        required.Test(excgrid::Component::TauA) || required.Test(excgrid::Component::TauB);
    const bool usesGradient = functional.UsesGradient();
    const bool needsHessians = usesGradient || usesTau;

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
    // density gradients or the kinetic-energy density - second derivatives per
    // batched point, plus that point's selection in the two flat lists the fetch
    // and the assembly both walk. The size is clamped to the largest block, so a
    // batch can never span blocks.
    const std::size_t bufferPoints =
        batch.pointBatch == 0 ? largestBlock : std::min(batch.pointBatch, largestBlock);
    const std::size_t aoCount = evaluator.AOCount();

    std::vector<double> batchValues(bufferPoints * aoCount);
    std::vector<double> batchGradients(bufferPoints * 3 * aoCount);
    std::vector<double> batchHessians(needsHessians ? bufferPoints * 6 * aoCount : 0);
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

                if (needsHessians)
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
                // Empty when neither sigma nor tau is read: the assembly below
                // must not read a buffer that was never allocated.
                const std::span<const double> hessians =
                    needsHessians ? std::span<const double>(
                                        batchHessians.data() + index * 6 * aoCount, 6 * aoCount)
                                  : std::span<const double>();
                const std::span<const std::size_t> selectedAos(
                    batchAos.data() + batchAoOffsets[index],
                    batchAoOffsets[index + 1] - batchAoOffsets[index]);
                const AoSlice aos{values, gradients, hessians, selectedAos};

                const PointField fieldA =
                    BuildPointField(densityAlpha, aos, contractions, usesGradient, usesTau);

                // The alpha sums read the contractions the alpha field was just
                // built from, so they are taken before the beta field overwrites
                // them. Nothing else about the point is spin-ordered.
                alphaSums.Clear();
                AccumulateSpinSums(alphaSums,
                                   aos,
                                   contractions,
                                   *atomOfAo,
                                   usesGradient,
                                   usesTau,
                                   evaluation.counts.aoDerivativeTerms);

                const PointField fieldB =
                    BuildPointField(densityBeta, aos, contractions, usesGradient, usesTau);

                betaSums.Clear();
                AccumulateSpinSums(betaSums,
                                   aos,
                                   contractions,
                                   *atomOfAo,
                                   usesGradient,
                                   usesTau,
                                   evaluation.counts.aoDerivativeTerms);

                const auto kernel =
                    EvaluateKernel("EvaluateXcGradient", functional, fieldA, fieldB);

                if (!kernel.has_value())
                {
                    return std::unexpected(kernel.error());
                }

                const double weight = batchWeights[index];
                evaluation.energy += weight * kernel->exc;

                // The weight's motion: the quadrature weight is a function of the
                // geometry, and the integrand rides it.
                const double* weightRow = weightFirst.data() + point * nuclearCoordinateCount;

                for (std::size_t coordinate = 0; coordinate < nuclearCoordinateCount; ++coordinate)
                {
                    evaluation.gradient[static_cast<Eigen::Index>(coordinate)] +=
                        kernel->exc * weightRow[coordinate];
                }

                // The point's motion: the integrand is a function of position, so
                // a point that moves with its atom carries its own value with it.
                const std::array<double, 3> energyDensityGradient =
                    GradientOfEnergyDensity(*kernel, fieldA, fieldB, usesGradient, usesTau);
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
                    std::span<double>(evaluation.gradient.data(), nuclearCoordinateCount),
                    alphaSums,
                    betaSums,
                    fieldA,
                    fieldB,
                    *kernel,
                    weight);
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

qcx::Result<XcGridMotionHessian> EvaluateXcGridMotionHessian(
    std::span<const excgrid::Block> blocks,
    const excgrid::GridDerivativeProvider& provider,
    const AoEvaluator& evaluator,
    const excgrid::XcFunctional& functional,
    std::span<const ShellEnvelope> envelopes,
    const XcIntegrationThresholds& thresholds,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta) {
    const std::string_view operation = "EvaluateXcGridMotionHessian";
    const Eigen::Index n = static_cast<Eigen::Index>(evaluator.AOCount());

    if (densityAlpha.rows() != n || densityAlpha.cols() != n || densityBeta.rows() != n ||
        densityBeta.cols() != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       std::string(operation) + ": both densities must be AOCount() x AOCount()"});
    }

    // A provider built for first order would refuse every block; naming the
    // requirement once, before any allocation, is what tells a caller what to
    // build rather than where it happened to notice.
    if (provider.MaxOrder() != excgrid::DerivativeOrder::kSecond)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       std::string(operation) +
                           ": the grid derivative provider must supply second-order derivatives"});
    }

    // The channel's own refusal, before any allocation: a functional whose
    // second response this walk cannot take is named here rather than answered
    // with part of one.
    if (const std::string refusal = ChannelRefusal(functional); !refusal.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented, std::string(operation) + ": " + refusal});
    }

    auto atomOfAo = BuildAtomOfAo(operation, envelopes, evaluator.AOCount());

    if (!atomOfAo.has_value())
    {
        return std::unexpected(atomOfAo.error());
    }

    std::size_t atomCount = 0;

    for (const ShellEnvelope& shell : envelopes)
    {
        atomCount = std::max(atomCount, shell.atomIndex + 1);
    }

    if (atomCount == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       std::string(operation) + ": the shell envelopes name no atom"});
    }

    std::size_t largestBlock = 0;

    for (const excgrid::Block& block : blocks)
    {
        largestBlock = std::max(largestBlock, block.pointCount);
    }

    const std::size_t nuclearCoordinateCount = 3U * atomCount;

    XcGridMotionHessian evaluation;
    evaluation.thresholds = thresholds;
    evaluation.gradient = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nuclearCoordinateCount));
    evaluation.hessian = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nuclearCoordinateCount),
                                               static_cast<Eigen::Index>(nuclearCoordinateCount));

    // What the functional reads decides both what the walk fetches and which
    // second response it owes. The rho fields' second response is expressible
    // with the value, gradient and Hessian tiers the evaluator already supplies,
    // so a functional that reads a density runs the electronic channel and the
    // walk fetches the AO Hessian for it even when it reads no density gradient
    // - a real cost rise against the first-order walk, which the
    // second-derivative call column makes visible.
    const excgrid::ComponentMask required = functional.RequiredMask();
    const bool usesTau =
        required.Test(excgrid::Component::TauA) || required.Test(excgrid::Component::TauB);
    const bool usesGradient = functional.UsesGradient();
    const bool electronicChannel = required.bits != 0;
    const bool needsHessians = usesGradient || usesTau || electronicChannel;
    const bool withSecond = usesGradient || electronicChannel;

    const std::size_t aoCount = evaluator.AOCount();

    PointContractions contractions;
    contractions.value.resize(aoCount);
    contractions.gradient.resize(3 * aoCount);
    SpinSums alphaSums(atomCount);
    SpinSums betaSums(atomCount);

    // One point at a time: the batch split exists so that one fetch serves a
    // batch of points, and here a fetch already serves the only point that
    // reads it. The scratch is sized for a point, not for a batch.
    std::vector<double> aoValues(aoCount);
    std::vector<double> aoGradients(3 * aoCount);
    std::vector<double> aoHessians(needsHessians ? 6 * aoCount : 0);
    std::vector<std::size_t> selectedShells(envelopes.size());
    std::vector<std::size_t> selectedAos(aoCount);
    // T_r, the integrand's total first derivative at fixed density: the point's
    // own motion plus the orbitals'.
    std::vector<double> pointMotion(nuclearCoordinateCount);

    // The electronic channel's scratch, reused across the grid. The components
    // its contraction runs over, in identifier order; G1 and the kernel's
    // contraction L, both flattened [nuclear coordinate][component]; the
    // right-hand sides one call carries; H_s X_r, [spin][coordinate][axis]; the
    // density partials' weighted AO curvature, [atom][row-major 3x3]; and the
    // kept atoms of the current point.
    std::vector<excgrid::Component> activeComponents;

    for (std::size_t index = 0; index < excgrid::kComponentCapacity; ++index)
    {
        const auto id = static_cast<excgrid::Component>(index);

        if (required.Test(id))
        {
            activeComponents.push_back(id);
        }
    }

    const std::size_t activeCount = activeComponents.size();
    // The tier's capacity is shared out between the active components, so one
    // call carries as many right-hand sides as fit and no more.
    const std::size_t contractionSides = std::max<std::size_t>(
        1, excgrid::kSecondDerivativeCapacity / std::max<std::size_t>(activeCount, 1));
    std::vector<double> firstResponse(nuclearCoordinateCount * activeCount);
    std::vector<double> curvatureContraction(nuclearCoordinateCount * activeCount);
    std::vector<double> rightHandSide(contractionSides * activeCount);
    std::vector<double> curvaturePosition(2 * nuclearCoordinateCount * 3);
    std::vector<double> curvaturePartial(atomCount * 9);
    std::vector<std::size_t> keptAtoms(atomCount);
    std::vector<char> atomKept(atomCount, 0);

    // The density's second-order orbital partials. Sized only when the channel
    // runs: a constant integrand must not pay for atomCount x AOCount storage it
    // never reads.
    SecondOrderSums alphaSecond(electronicChannel ? atomCount : 0, electronicChannel ? aoCount : 0);
    SecondOrderSums betaSecond(electronicChannel ? atomCount : 0, electronicChannel ? aoCount : 0);

    // The provider's buffers, sized once for the largest block, at the order the
    // second derivative needs: the curvature arrays are the whole point of this
    // walk and the first-order ones are read from the same request.
    excgrid::DerivativeRequest request;
    request.parts = BothParts();
    request.order = excgrid::DerivativeOrder::kSecond;
    request.nuclearCoordinateCount = nuclearCoordinateCount;

    const excgrid::BlockDerivativeSizes providerSizes = excgrid::SizeOf(request, largestBlock);
    std::vector<double> positionFirst(providerSizes.positionFirst);
    std::vector<double> weightFirst(providerSizes.weightFirst);
    std::vector<double> positionSecond(providerSizes.positionSecond);
    std::vector<double> weightSecond(providerSizes.weightSecond);

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
                           std::string(operation) +
                               ": a block's points, weights and owning atoms must each hold its "
                               "point count entries"});
        }

        const excgrid::BlockDerivativeSizes sizes = excgrid::SizeOf(request, block.pointCount);
        excgrid::BlockDerivatives derivatives;
        derivatives.positionFirst = std::span<double>(positionFirst.data(), sizes.positionFirst);
        derivatives.weightFirst = std::span<double>(weightFirst.data(), sizes.weightFirst);
        derivatives.positionSecond = std::span<double>(positionSecond.data(), sizes.positionSecond);
        derivatives.weightSecond = std::span<double>(weightSecond.data(), sizes.weightSecond);
        derivatives.atomIndex = block.atomIndex;

        const excgrid::DerivativeStatus status = provider.Evaluate(block, request, derivatives);

        if (status != excgrid::DerivativeStatus::kOk)
        {
            return std::unexpected(qcx::Error{
                qcx::ErrorCode::kInvalidArgument,
                std::string(operation) + ": the grid derivative provider refused a block: " +
                    std::string(DerivativeStatusName(status))});
        }

        ++evaluation.counts.batches;
        evaluation.counts.weightDerivativeTerms += sizes.weightFirst;
        evaluation.counts.coordinateDerivativeTerms += sizes.positionFirst;
        evaluation.counts.weightCurvatureTerms += sizes.weightSecond;
        evaluation.counts.coordinateCurvatureTerms += sizes.positionSecond;

        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            const double weight = block.weights[point];

            if (weight == 0.0)
            {
                continue;
            }

            ++evaluation.counts.points;

            // The selection the energy path makes at this point, under the same
            // tolerance and from the same weights.
            std::size_t selectedShellCount = 0;
            std::size_t selectedAoCount = 0;

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
                selectedShells[selectedShellCount++] = shell;

                for (std::size_t ao = 0; ao < envelopes[shell].functionCount; ++ao)
                {
                    selectedAos[selectedAoCount++] = envelopes[shell].aoOffset + ao;
                }
            }

            if (selectedShellCount == 0)
            {
                continue;
            }

            evaluation.counts.aoSlotsEvaluated += selectedAoCount;
            evaluation.counts.contractionPairs += 2 * selectedAoCount * selectedAoCount;
            ++evaluation.counts.aoFetches;

            const std::span<const std::size_t> selectionShells(selectedShells.data(),
                                                               selectedShellCount);

            if (needsHessians)
            {
                evaluator.EvaluateDerivativesSelected(
                    block.points[point],
                    selectionShells,
                    std::span<double>(aoValues.data(), aoCount),
                    std::span<double>(aoGradients.data(), 3 * aoCount),
                    std::span<double>(aoHessians.data(), 6 * aoCount));
            } else
            {
                evaluator.EvaluateGradientsSelected(
                    block.points[point],
                    selectionShells,
                    std::span<double>(aoValues.data(), aoCount),
                    std::span<double>(aoGradients.data(), 3 * aoCount));
            }

            // Empty when nothing needs it: BuildPointField, the spin sums and
            // the density channel must not read a buffer that was never
            // allocated.
            const std::span<const double> hessians =
                needsHessians ? std::span<const double>(aoHessians.data(), 6 * aoCount)
                              : std::span<const double>();
            const AoSlice aos{std::span<const double>(aoValues.data(), aoCount),
                              std::span<const double>(aoGradients.data(), 3 * aoCount),
                              hessians,
                              std::span<const std::size_t>(selectedAos.data(), selectedAoCount)};

            // The atoms this point kept. The density channel's second response
            // is accumulated in atom blocks, so this is the list its sums are
            // cleared and read over; the flags are cleared with it, so the next
            // point starts from an empty list.
            std::size_t keptAtomCount = 0;

            if (electronicChannel)
            {
                for (const std::size_t ao : aos.selectedAos)
                {
                    const std::size_t atom = (*atomOfAo)[ao];

                    if (atomKept[atom] == 0)
                    {
                        atomKept[atom] = 1;
                        keptAtoms[keptAtomCount++] = atom;
                    }
                }

                const std::span<const std::size_t> atoms(keptAtoms.data(), keptAtomCount);
                ClearSecondOrderSums(alphaSecond, aos, atoms);
                ClearSecondOrderSums(betaSecond, aos, atoms);

                for (std::size_t index = 0; index < keptAtomCount; ++index)
                {
                    atomKept[keptAtoms[index]] = 0;
                }
            }

            const PointField fieldA =
                BuildPointField(densityAlpha, aos, contractions, withSecond, usesTau);

            alphaSums.Clear();
            AccumulateSpinSums(alphaSums,
                               aos,
                               contractions,
                               *atomOfAo,
                               withSecond,
                               usesTau,
                               evaluation.counts.aoDerivativeTerms);

            if (electronicChannel)
            {
                // The contractions still belong to the alpha field, so this
                // reads them before the beta field overwrites them.
                AccumulateSecondOrderSums(alphaSecond, aos, contractions, densityAlpha, *atomOfAo);
            }

            const PointField fieldB =
                BuildPointField(densityBeta, aos, contractions, withSecond, usesTau);

            betaSums.Clear();
            AccumulateSpinSums(betaSums,
                               aos,
                               contractions,
                               *atomOfAo,
                               withSecond,
                               usesTau,
                               evaluation.counts.aoDerivativeTerms);

            if (electronicChannel)
            {
                AccumulateSecondOrderSums(betaSecond, aos, contractions, densityBeta, *atomOfAo);
            }

            const auto kernel = EvaluateKernel(operation, functional, fieldA, fieldB);

            if (!kernel.has_value())
            {
                return std::unexpected(kernel.error());
            }

            evaluation.energy += weight * kernel->exc;

            const std::array<double, 3> energyDensityGradient =
                GradientOfEnergyDensity(*kernel, fieldA, fieldB, usesGradient, usesTau);

            // T_r: the integrand's total derivative through both the point's own
            // motion and the orbitals'. The gradient walk accumulates the two
            // separately because each multiplies a different weight derivative;
            // here each weight derivative multiplies the whole sum, so it is
            // formed once. The orbital half is the same function at unit weight.
            std::fill(pointMotion.begin(), pointMotion.end(), 0.0);
            AccumulateOrbitalMotion(std::span<double>(pointMotion.data(), nuclearCoordinateCount),
                                    alphaSums,
                                    betaSums,
                                    fieldA,
                                    fieldB,
                                    *kernel,
                                    1.0);

            const double* positionRow = positionFirst.data() + point * 3 * nuclearCoordinateCount;

            for (std::size_t direction = 0; direction < 3; ++direction)
            {
                const double* row = positionRow + direction * nuclearCoordinateCount;

                for (std::size_t coordinate = 0; coordinate < nuclearCoordinateCount; ++coordinate)
                {
                    pointMotion[coordinate] += energyDensityGradient[direction] * row[coordinate];
                }
            }

            // The electronic channel: the integrand's second response at fixed
            // density. For a functional that reads only the spin densities the
            // chain rule over the fields is
            //
            //     D_qD_r e = sum_i F_i D_qD_r u_i + sum_ij F_ij (D_q u_i)(D_r u_j),
            //
            // with u the two densities, F the kernel's partials and
            //
            //     D_q u_s = X_q . grad rho_s - 2 P^s_A(q)
            //
            // the field's total first response: the point's own motion plus the
            // orbitals'. The density's second response is
            //
            //     D_qD_r rho_s = X_q^T (hess rho_s) X_r - 2 X_q . Q^s_B[b, :]
            //                    - 2 X_r . Q^s_A[a, :] + 2 C^s_{A,B}[a][b]
            //                    + 2 U^s[e] on the diagonal atom block,
            //
            // where C pairs two atoms and U exists only where they coincide, and
            // every term is a value, gradient or Hessian tier away, which is why
            // this channel needs no new AO derivative. The position curvature
            // X^c_,qr is deliberately absent: the walk's own triangle below
            // carries it, once, and adding it here would double it.
            if (electronicChannel)
            {
                const std::array<const PointField*, 2> fields = {&fieldA, &fieldB};
                const std::array<const SpinSums*, 2> spinSums = {&alphaSums, &betaSums};
                const std::array<const SecondOrderSums*, 2> secondSums = {&alphaSecond,
                                                                          &betaSecond};
                // A component the functional does not read has no partial, and
                // its field is not in G1 either, so zeroing the unused slot is
                // the same as never reading it.
                const std::array<double, 2> partials = {
                    required.Test(excgrid::Component::RhoA) ? kernel->vrhoA : 0.0,
                    required.Test(excgrid::Component::RhoB) ? kernel->vrhoB : 0.0};

                // G1, one entry per active component per nuclear coordinate.
                for (std::size_t coordinate = 0; coordinate < nuclearCoordinateCount; ++coordinate)
                {
                    const std::size_t atom = coordinate / 3;
                    const std::size_t direction = coordinate % 3;

                    for (std::size_t slot = 0; slot < activeCount; ++slot)
                    {
                        const std::size_t spin = SpinOfDensity(activeComponents[slot]);
                        const PointField& field = *fields[spin];
                        double response = -2.0 * (*spinSums[spin]).position[atom][direction];

                        for (std::size_t axis = 0; axis < 3; ++axis)
                        {
                            response += positionRow[axis * nuclearCoordinateCount + coordinate] *
                                        field.gradient[axis];
                        }

                        firstResponse[coordinate * activeCount + slot] = response;
                    }
                }

                // The kernel's curvature, contracted against G1 in chunks the
                // tier's capacity holds. The inputs carry exactly the required
                // components: the contraction's width and the right-hand side's
                // layout are both read from this mask, so a wider one would
                // misread the right-hand sides.
                excgrid::PointInputs curvatureInputs;

                for (const excgrid::Component id : activeComponents)
                {
                    curvatureInputs.Set(id, fields[SpinOfDensity(id)]->rho);
                }

                for (std::size_t start = 0; start < nuclearCoordinateCount;
                     start += contractionSides)
                {
                    const std::size_t count =
                        std::min(contractionSides, nuclearCoordinateCount - start);

                    for (std::size_t side = 0; side < count; ++side)
                    {
                        for (std::size_t slot = 0; slot < activeCount; ++slot)
                        {
                            rightHandSide[side * activeCount + slot] =
                                firstResponse[(start + side) * activeCount + slot];
                        }
                    }

                    // The result the tier also fills is the kernel value the
                    // walk already holds, so it is read from nowhere.
                    excgrid::PointResult redundantResult;
                    excgrid::PointSecondDerivative contracted;
                    const excgrid::KernelStatus status =
                        functional.EvaluatePointWithSecondDerivatives(
                            curvatureInputs,
                            std::span<const double>(rightHandSide.data(), count * activeCount),
                            redundantResult,
                            contracted);

                    if (status != excgrid::KernelStatus::kOk)
                    {
                        return std::unexpected(
                            qcx::Error{qcx::ErrorCode::kUnimplemented,
                                       std::string(operation) + ": the functional " +
                                           std::string(excgrid::DescribeStatus(status))});
                    }

                    // The refusal above is the functional's; this one is the
                    // library answering a different question from the one asked,
                    // which no finite difference would attribute to the kernel.
                    // The mask is compared whole rather than by its population:
                    // the contraction's layout is this walk's active order, so a
                    // different set of the same size would be read wrongly.
                    if (contracted.mask.bits != curvatureInputs.mask.bits ||
                        contracted.rightHandSides != count)
                    {
                        return std::unexpected(qcx::Error{
                            qcx::ErrorCode::kInternalError,
                            std::string(operation) +
                                ": the second-derivative contraction answered for a different "
                                "component set or a different right-hand side count"});
                    }

                    for (std::size_t side = 0; side < count; ++side)
                    {
                        for (std::size_t slot = 0; slot < activeCount; ++slot)
                        {
                            curvatureContraction[(start + side) * activeCount + slot] =
                                contracted.contracted[side * activeCount + slot];
                        }
                    }

                    ++evaluation.counts.kernelSecondDerivativeCalls;
                }

                const std::span<const std::size_t> atoms(keptAtoms.data(), keptAtomCount);

                // H_s X_r, per spin and nuclear coordinate: the position part of
                // the density's second response is one quadratic form in these.
                for (std::size_t spin = 0; spin < 2; ++spin)
                {
                    const PointField& field = *fields[spin];
                    double* target = curvaturePosition.data() + spin * 3 * nuclearCoordinateCount;

                    for (std::size_t coordinate = 0; coordinate < nuclearCoordinateCount;
                         ++coordinate)
                    {
                        for (std::size_t rowAxis = 0; rowAxis < 3; ++rowAxis)
                        {
                            double value = 0.0;

                            for (std::size_t columnAxis = 0; columnAxis < 3; ++columnAxis)
                            {
                                value +=
                                    field.second[3 * rowAxis + columnAxis] *
                                    positionRow[columnAxis * nuclearCoordinateCount + coordinate];
                            }

                            target[3 * coordinate + rowAxis] = value;
                        }
                    }
                }

                // The curvature's partials, the two spins' weighted by their
                // density partials: every cross term of the density's second
                // response reads one atom's block.
                for (const std::size_t atom : atoms)
                {
                    for (std::size_t entry = 0; entry < 9; ++entry)
                    {
                        curvaturePartial[atom * 9 + entry] =
                            partials[0] * alphaSums.curvature[atom][entry] +
                            partials[1] * betaSums.curvature[atom][entry];
                    }
                }

                // theta_q theta_r rho, the pair half:
                //   2 sum_{mu,nu} D_mu nu grad phi_mu[a] grad phi_nu[b].
                // It pairs two atoms, so it is scattered straight into the
                // triangle from the blocked sums rather than through a
                // point-sized pair array. Each spin's pair reads its own block,
                // which is why the two blocked sums are weighted and added.
                for (const std::size_t atomB : atoms)
                {
                    for (std::size_t directionB = 0; directionB < 3; ++directionB)
                    {
                        for (const std::size_t mu : aos.selectedAos)
                        {
                            const double blocked =
                                partials[0] * secondSums[0]->Blocked(atomB, directionB, mu) +
                                partials[1] * secondSums[1]->Blocked(atomB, directionB, mu);

                            if (blocked == 0.0)
                            {
                                continue;
                            }

                            const std::size_t atomA = (*atomOfAo)[mu];

                            for (std::size_t directionA = 0; directionA < 3; ++directionA)
                            {
                                const std::size_t rowIndex = 3 * atomA + directionA;
                                const std::size_t columnIndex = 3 * atomB + directionB;

                                if (rowIndex > columnIndex)
                                {
                                    continue;
                                }

                                evaluation.hessian(static_cast<Eigen::Index>(rowIndex),
                                                   static_cast<Eigen::Index>(columnIndex)) +=
                                    weight * 2.0 * aos.gradients[3 * mu + directionA] * blocked;
                            }
                        }
                    }
                }

                // theta_q theta_r rho, the curvature half:
                //   2 sum_{mu in A} (D phi)_mu hess phi_mu[a][b].
                for (const std::size_t atom : atoms)
                {
                    const std::array<double, 9>& curvatureA = alphaSecond.curvature[atom];
                    const std::array<double, 9>& curvatureB = betaSecond.curvature[atom];

                    for (std::size_t rowAxis = 0; rowAxis < 3; ++rowAxis)
                    {
                        for (std::size_t columnAxis = rowAxis; columnAxis < 3; ++columnAxis)
                        {
                            const std::size_t entry = 3 * rowAxis + columnAxis;
                            const double curvature =
                                partials[0] * curvatureA[entry] + partials[1] * curvatureB[entry];

                            evaluation.hessian(static_cast<Eigen::Index>(3 * atom + rowAxis),
                                               static_cast<Eigen::Index>(3 * atom + columnAxis)) +=
                                weight * 2.0 * curvature;
                        }
                    }
                }

                // What is left of the density's second response, one triangle
                // entry at a time:
                //     X_q^T (sum_s F_s hess rho_s) X_r - 2 X_q . Q_B[b, :]
                //                                    - 2 X_r . Q_A[a, :]
                //                                    + sum_i L_i[q] G1_i[r].
                for (std::size_t rowIndex = 0; rowIndex < nuclearCoordinateCount; ++rowIndex)
                {
                    const std::size_t atomA = rowIndex / 3;
                    const std::size_t directionA = rowIndex % 3;
                    const double* xRow = positionRow + rowIndex;

                    for (std::size_t columnIndex = rowIndex; columnIndex < nuclearCoordinateCount;
                         ++columnIndex)
                    {
                        const std::size_t atomB = columnIndex / 3;
                        const std::size_t directionB = columnIndex % 3;
                        const double* xColumn = positionRow + columnIndex;
                        double value = 0.0;

                        for (std::size_t axis = 0; axis < 3; ++axis)
                        {
                            const double alpha = curvaturePosition[3 * columnIndex + axis];
                            const double beta = curvaturePosition[3 * nuclearCoordinateCount +
                                                                  3 * columnIndex + axis];
                            value += xRow[axis * nuclearCoordinateCount] *
                                     (partials[0] * alpha + partials[1] * beta);
                        }

                        for (std::size_t axis = 0; axis < 3; ++axis)
                        {
                            value -= 2.0 * xRow[axis * nuclearCoordinateCount] *
                                     curvaturePartial[atomB * 9 + 3 * directionB + axis];
                            value -= 2.0 * xColumn[axis * nuclearCoordinateCount] *
                                     curvaturePartial[atomA * 9 + 3 * directionA + axis];
                        }

                        for (std::size_t slot = 0; slot < activeCount; ++slot)
                        {
                            value += curvatureContraction[rowIndex * activeCount + slot] *
                                     firstResponse[columnIndex * activeCount + slot];
                        }

                        evaluation.hessian(static_cast<Eigen::Index>(rowIndex),
                                           static_cast<Eigen::Index>(columnIndex)) +=
                            weight * value;
                    }
                }

                evaluation.counts.electronicCurvatureTerms += keptAtomCount * keptAtomCount * 9;
            }

            // The gradient this walk owes the total: the weight's motion and the
            // integrand's, which is the whole of the first-order result.
            const double* weightRow = weightFirst.data() + point * nuclearCoordinateCount;

            for (std::size_t coordinate = 0; coordinate < nuclearCoordinateCount; ++coordinate)
            {
                evaluation.gradient[static_cast<Eigen::Index>(coordinate)] +=
                    weightRow[coordinate] * kernel->exc + weight * pointMotion[coordinate];
            }

            //     d2(W e)/dR_q dR_r = W_,qr e + W_,q T_r + W_,r T_q
            //                       + W sum_c (de/x^c) X^c_,qr + W M_qr ,
            // with M the electronic channel accumulated just above. This is the
            // derivative of the accumulated gradient with the integrand's own
            // response split out, and the three pieces are disjoint: the weight
            // terms differentiate the weight, the position-curvature term
            // differentiates the fields' first derivatives, and M holds the rest.
            const std::size_t curvatureRow = nuclearCoordinateCount * nuclearCoordinateCount;
            const double* weightCurvature = weightSecond.data() + point * curvatureRow;
            const double* positionCurvature = positionSecond.data() + point * 3 * curvatureRow;

            for (std::size_t rowIndex = 0; rowIndex < nuclearCoordinateCount; ++rowIndex)
            {
                for (std::size_t column = rowIndex; column < nuclearCoordinateCount; ++column)
                {
                    const std::size_t entry = rowIndex * nuclearCoordinateCount + column;
                    double value = weightCurvature[entry] * kernel->exc +
                                   weightRow[rowIndex] * pointMotion[column] +
                                   weightRow[column] * pointMotion[rowIndex];

                    for (std::size_t direction = 0; direction < 3; ++direction)
                    {
                        value += weight * energyDensityGradient[direction] *
                                 positionCurvature[direction * curvatureRow + entry];
                    }

                    evaluation.hessian(static_cast<Eigen::Index>(rowIndex),
                                       static_cast<Eigen::Index>(column)) += value;
                }
            }
        }
    }

    // One triangle was accumulated, so the mirror is exact rather than an
    // agreement to rounding between two sums.
    for (std::size_t rowIndex = 0; rowIndex < nuclearCoordinateCount; ++rowIndex)
    {
        for (std::size_t column = rowIndex + 1; column < nuclearCoordinateCount; ++column)
        {
            evaluation.hessian(static_cast<Eigen::Index>(column),
                               static_cast<Eigen::Index>(rowIndex)) =
                evaluation.hessian(static_cast<Eigen::Index>(rowIndex),
                                   static_cast<Eigen::Index>(column));
        }
    }

    evaluation.counts.aoSlotsDense = evaluation.counts.points * aoCount;
    evaluation.counts.contractionPairsDense = 2 * evaluation.counts.points * aoCount * aoCount;
    evaluation.counts.aoDerivativeTermsDense = 2 * evaluation.counts.points * aoCount;
    evaluation.counts.electronicCurvatureTermsDense =
        evaluation.counts.points * atomCount * atomCount * 9;

    return evaluation;
}

qcx::Result<XcGridMotionHessian> EvaluateXcGridMotionHessian(
    const XcGridEngine& engine,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    const excgrid::GridDerivativeProvider& provider,
    const XcIntegrationThresholds& thresholds) {
    return EvaluateXcGridMotionHessian(engine.Grid().Blocks(),
                                       provider,
                                       engine.Evaluator(),
                                       engine.Functional(),
                                       engine.ShellEnvelopes(),
                                       thresholds,
                                       densityAlpha,
                                       densityBeta);
}

} // namespace qcx::grid
