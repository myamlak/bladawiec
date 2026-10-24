// The two-electron ERI gradient, at fixed density (two_electron.hpp): the
// nuclear-coordinate derivative of E = 1/2 sum_uvls D_uv D_ls (uv|ls).
//
// The walk enumerates the canonical shell pairs and, for each unordered pair
// of them, one shell quartet. The ordered shell sum the energy carries is
// recovered from the canonical walk by the pair multiplicities: a pair
// (i, j) with i != j stands for two ordered pairs, and a pair pair (P, Q)
// with P != Q for two ordered pairs of pairs - the density matrix is
// symmetric, so both orientations carry the same block. Written out,
//
//   E = 1/2 sum_P m_P^2 D_P^2 (P|P) + sum_{P<Q} m_P m_Q D_P D_Q (P|Q),
//
// with m_P = 2 for an off-diagonal pair and 1 for a diagonal one.
//
// Each quartet's block is differentiated through the two-electron derivative
// tier (md_eri_derivative.hpp) and contracted with the density before any
// accumulation, so the largest object the walk holds is one quartet's block
// and no four-index tensor is formed at any point. The energy is accumulated
// from the same value blocks and the same weights, which is what makes the
// reported energy the functional the reported gradient differentiates.
// [Helgaker2000] [McMurchie1978]

#include "internal/fock_screen.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_eri_derivative.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/integrals/two_electron.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace qcx::integrals {

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::integrals::internal::MdEriDerivativeQuartet;
using qcx::integrals::internal::MdEriDerivativeScratch;

/// The distinct atoms a quartet's four shells sit on, three flat coordinates
/// (3 * atom + axis) each, in ascending atom order. A coordinate on an atom
/// the quartet omits contributes an exactly zero block, so requesting the
/// atoms the quartet actually carries is the whole of it.
/// \param atoms The quartet's four atoms.
/// \param coordinates Out: the flat coordinates.
void QuartetCoordinates(const std::array<std::size_t, 4>& atoms,
                        std::vector<std::size_t>& coordinates) {
    std::array<std::size_t, 4> distinct = atoms;
    std::sort(distinct.begin(), distinct.end());
    const auto last = std::unique(distinct.begin(), distinct.end());
    coordinates.clear();

    for (auto atom = distinct.begin(); atom != last; ++atom)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            coordinates.push_back(3 * (*atom) + static_cast<std::size_t>(axis));
        }
    }
}

/// The ordered shell-pair multiplicity of one canonical pair: 2 when the two
/// shells differ, 1 when they coincide. The canonical walk counts each
/// unordered pair once, and the ordered sum the functional carries counts an
/// off-diagonal pair twice.
/// \param pair The canonical pair.
/// \returns 1 or 2.
std::size_t PairMultiplicity(const ShellPairIndex& pair) noexcept {
    return pair.i == pair.j ? 1 : 2;
}

/// The dot product of two equal-length blocks.
/// \param left The first block.
/// \param right The second block.
/// \returns sum_i left[i] * right[i].
double DotProduct(std::span<const double> left, std::span<const double> right) noexcept {
    double sum = 0.0;

    for (std::size_t index = 0; index < left.size(); ++index)
    {
        sum += left[index] * right[index];
    }

    return sum;
}

/// Checks every shell of the pair list against the engine's angular-momentum
/// cap.
/// \param pairList The flattened shell list.
/// \returns Nothing, or kUnimplemented for a shell past kMaxEngineL.
qcx::Result<void> CheckShells(const ShellPairList& pairList) {
    for (const ShellInfo& shell : pairList.shells)
    {
        if (!SupportsL(shell.angularMomentum))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "shell angular momentum exceeds kMaxEngineL of this build"});
        }
    }

    return {};
}

} // namespace

qcx::Result<EriGradient> ComputeEriGradient(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet,
                                            const CpuTensor2& density,
                                            const EriGradientOptions& options) {
    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto shellsChecked = CheckShells(*pairList);

    if (!shellsChecked.has_value())
    {
        return std::unexpected(shellsChecked.error());
    }

    const std::size_t n = pairList->functionCount;

    if (density.Shape()[0] != n || density.Shape()[1] != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the density matrix is not nBasis x nBasis for this basis set"});
    }

    auto shells = internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    const std::size_t nAtoms = molecule.AtomCount();
    EriGradient result;
    result.gradient = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(3 * nAtoms));

    if (n == 0 || pairList->pairs.empty())
    {
        return result;
    }

    Eigen::MatrixXd densityMatrix(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            densityMatrix(static_cast<Eigen::Index>(u), static_cast<Eigen::Index>(v)) =
                density(u, v);
        }
    }

    // The screen's pair bounds and density maxima, and the threshold they are
    // read against. The threshold is the energy path's own: the preset's
    // Schwarz threshold with the neighbor-list slack's margin, so the screen
    // can only keep quartets an energy walk at the same preset keeps.
    std::vector<TwoElectronPairBound> bounds;
    std::vector<double> pairDensity;
    double threshold = 0.0;

    if (options.useDensityScreening)
    {
        auto computedBounds = ComputeTwoElectronPairBounds(molecule, basisSet);

        if (!computedBounds.has_value())
        {
            return std::unexpected(computedBounds.error());
        }

        bounds = std::move(*computedBounds);
        pairDensity = internal::BuildShellPairMaxDensity(densityMatrix, *pairList);
        threshold = SchwarzThreshold(options.accuracy) / (1.0 + internal::kNeighborListSlack);
    }

    MdEriDerivativeScratch scratch;
    std::vector<std::size_t> coordinates;
    std::vector<double> value;
    std::vector<double> derivatives;
    std::vector<double> weights;
    const std::size_t pairCount = pairList->pairs.size();

    for (std::size_t braIndex = 0; braIndex < pairCount; ++braIndex)
    {
        const ShellPairIndex& braPair = pairList->pairs[braIndex];
        const ShellInfo& braI = pairList->shells[braPair.i];
        const ShellInfo& braJ = pairList->shells[braPair.j];
        const std::size_t nI = ShellFunctionCount(braI);
        const std::size_t nJ = ShellFunctionCount(braJ);
        const std::size_t braMultiplicity = PairMultiplicity(braPair);
        ++result.counts.braPairs;

        for (std::size_t ketIndex = 0; ketIndex <= braIndex; ++ketIndex)
        {
            ++result.counts.ketPairs;
            const ShellPairIndex& ketPair = pairList->pairs[ketIndex];
            const ShellInfo& ketK = pairList->shells[ketPair.i];
            const ShellInfo& ketL = pairList->shells[ketPair.j];
            const std::size_t nK = ShellFunctionCount(ketK);
            const std::size_t nL = ShellFunctionCount(ketL);
            const std::size_t ketMultiplicity = PairMultiplicity(ketPair);
            const std::size_t pairPairFactor = braIndex == ketIndex ? 1 : 2;
            const double permutationWeight =
                0.5 * static_cast<double>(pairPairFactor * braMultiplicity * ketMultiplicity);

            if (options.useDensityScreening)
            {
                const QuartetDerivativeBound bound =
                    QuartetDerivativeProduct(bounds[braIndex], bounds[ketIndex]);
                const double scaled =
                    permutationWeight * pairDensity[braIndex] * pairDensity[ketIndex];

                if (scaled * std::max(bound.value, bound.derivative) < threshold)
                {
                    ++result.counts.screenedQuartets;
                    continue;
                }
            }

            MdEriDerivativeQuartet quartet;
            quartet.shells = {&(*shells)[braPair.i],
                              &(*shells)[braPair.j],
                              &(*shells)[ketPair.i],
                              &(*shells)[ketPair.j]};
            quartet.atoms = {braI.atomIndex, braJ.atomIndex, ketK.atomIndex, ketL.atomIndex};
            QuartetCoordinates(quartet.atoms, coordinates);

            const std::size_t elements = internal::EriQuartetBlockElements(quartet);
            derivatives.assign(coordinates.size() * elements, 0.0);

            // The derivative build runs first: it is the call that refuses a
            // quartet whose Hermite order the tier cannot reach, and the value
            // build that follows it is safe once the request has passed.
            auto differentiated = internal::BuildEriQuartetDerivative(
                quartet, 1, coordinates, derivatives, scratch, nAtoms);

            if (!differentiated.has_value())
            {
                return std::unexpected(differentiated.error());
            }

            value.assign(elements, 0.0);
            internal::BuildEriQuartetValue(quartet, 1, value, scratch);

            weights.assign(elements, 0.0);

            for (std::size_t fa = 0; fa < nI; ++fa)
            {
                for (std::size_t fb = 0; fb < nJ; ++fb)
                {
                    const double densityBra =
                        densityMatrix(static_cast<Eigen::Index>(braI.functionOffset + fa),
                                      static_cast<Eigen::Index>(braJ.functionOffset + fb));

                    for (std::size_t fc = 0; fc < nK; ++fc)
                    {
                        for (std::size_t fd = 0; fd < nL; ++fd)
                        {
                            const double densityKet =
                                densityMatrix(static_cast<Eigen::Index>(ketK.functionOffset + fc),
                                              static_cast<Eigen::Index>(ketL.functionOffset + fd));
                            weights[EriBlockIndex(fa, fb, fc, fd, nI, nJ, nK, nL)] =
                                permutationWeight * densityBra * densityKet;
                        }
                    }
                }
            }

            result.energy += DotProduct(weights, value);

            for (std::size_t coordinate = 0; coordinate < coordinates.size(); ++coordinate)
            {
                const std::span<const double> block =
                    std::span<const double>(derivatives).subspan(coordinate * elements, elements);
                result.gradient(static_cast<Eigen::Index>(coordinates[coordinate])) +=
                    DotProduct(weights, block);
            }

            ++result.counts.quartets;
        }
    }

    return result;
}

} // namespace qcx::integrals
