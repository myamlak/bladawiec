#pragma once

/// \file
/// The derivative machinery the RI Coulomb entries share: the phantom-s
/// quartet representation, the floored metric inverse as a reusable linear
/// operator, the one retained task set both the energy and the derivatives
/// read, and the workers that accumulate d, d^(U) and w^(U).
///
/// A tuple is `order` flat coordinates, 3 * atom + axis, in request order.
/// Sub-tuples are enumerated by positions, so a repeated coordinate in one
/// tuple (a diagonal second derivative) is two distinct positions and the
/// tier's own tuple handling carries the repetition.

#include "internal/md_batch.hpp"
#include "internal/md_eri_derivative.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace qcx::integrals::internal {

/// The phantom s function an auxiliary shell is paired with: angular
/// momentum 0, exponent 0 and unit coefficient, at the auxiliary shell's own
/// centre. Exponent 0 makes the pair's second function the constant 1, so
/// the pair reproduces the bare Coulomb operator the resolution of identity
/// fits against - the same phantom the energy path's auxiliary pair data
/// carries.
/// \param center The auxiliary shell's centre (Bohr).
/// \returns The phantom shell input.
inline MdShellInput PhantomShell(const std::array<double, 3>& center) {
    MdShellInput shell;
    shell.contractions.angularMomentum = 0;
    shell.contractions.isSpherical = true;
    shell.contractions.rows = 1;
    shell.contractions.exponents = {0.0};
    shell.contractions.normalized = {{1.0}};
    shell.cx = center[0];
    shell.cy = center[1];
    shell.cz = center[2];
    return shell;
}

/// The distinct atoms a quartet's four shells sit on, three axes each, in
/// ascending atom order - the coordinates the quartet's blocks are built
/// for. A shell on an atom the list omits has no operator there, so
/// requesting the atoms the quartet actually carries is the whole of it.
/// \param atoms The quartet's four atoms.
/// \param coordinates Out: the flat coordinates.
inline void QuartetCoordinates(const std::array<std::size_t, 4>& atoms,
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
inline std::size_t PairMultiplicity(const ShellPairIndex& pair) noexcept {
    return pair.i == pair.j ? 1 : 2;
}

/// The largest magnitude the fitting coefficients carry over one auxiliary
/// shell's functions - the weight a screen has to bound a whole shell by.
/// \param coefficients The coefficients, one per auxiliary function.
/// \param offset The shell's first auxiliary function.
/// \param nFuncs The shell's function count.
/// \returns The largest |w| over the shell.
inline double AuxiliaryWeightMax(const Eigen::VectorXd& coefficients,
                                 std::size_t offset,
                                 std::size_t nFuncs) {
    double largest = 0.0;

    for (std::size_t f = 0; f < nFuncs; ++f)
    {
        largest = std::max(largest, std::abs(coefficients(static_cast<Eigen::Index>(offset + f))));
    }

    return largest;
}

/// The auxiliary metric's floored eigen-inverse, factored once and applied as
/// a fixed linear operator: the energy path's solve at the same floor, so the
/// coefficients the derivatives differentiate are the coefficients the
/// energy used. Init keeps the order-1 entry's two error messages verbatim.
struct FlooredMetricInverse {
    Eigen::MatrixXd eigenvectors; ///< The metric's eigenvectors, columns.
    Eigen::VectorXd inverseEigenvalues; ///< 1/lambda above the floor, 0 below.

    /// Factors one metric at one floor.
    /// \param metric The auxiliary metric, nAux x nAux, symmetric.
    /// \param floorEpsilon The relative eigenvalue floor.
    /// \returns Nothing, or an Error (kInvalidArgument) when the metric fails
    /// its eigendecomposition or is degenerate below the floor.
    qcx::Result<void> Init(const Eigen::MatrixXd& metric, double floorEpsilon) {
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(metric);

        if (solver.info() != Eigen::Success)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "the auxiliary-basis metric failed its eigendecomposition"});
        }

        const Eigen::VectorXd& eigenvalues = solver.eigenvalues();
        const double lambdaMax = eigenvalues.maxCoeff();
        const double floor = floorEpsilon * lambdaMax;
        inverseEigenvalues.resize(eigenvalues.size());

        for (Eigen::Index i = 0; i < eigenvalues.size(); ++i)
        {
            inverseEigenvalues(i) = eigenvalues(i) >= floor ? 1.0 / eigenvalues(i) : 0.0;
        }

        if (lambdaMax <= 0.0 || !(inverseEigenvalues.array() > 0.0).any())
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "the auxiliary-basis metric is degenerate below the RI-J floor"});
        }

        eigenvectors = solver.eigenvectors();
        return {};
    }

    /// Applies the floored inverse.
    /// \param vector The right-hand side.
    /// \returns M^-1 vector through the floored factorization.
    Eigen::VectorXd Apply(const Eigen::VectorXd& vector) const {
        return eigenvectors * inverseEigenvalues.cwiseProduct(eigenvectors.transpose() * vector);
    }
};

/// The one retained task set the energy and every derivative read: the
/// energy path's density-free predicate unioned with the derivative-aware
/// one, plus the threshold both read.
struct RiTaskScreen {
    bool active = false; ///< Off: every task is retained (the exact walk).
    double threshold = 0.0; ///< The preset's Schwarz threshold with the slack.
    std::vector<TwoElectronPairBound> pairBounds; ///< One per canonical orbital pair.
    std::vector<TwoElectronPairBound> auxDiagonalBounds; ///< One per auxiliary shell.
    std::vector<double> pairDensity; ///< One per canonical orbital pair.
    std::vector<double> auxWeightMax; ///< One per auxiliary shell, from d's solve.

    /// The retention rule: the union of the energy path's density-free
    /// predicate and the derivative-aware one.
    /// \param pairIndex The canonical orbital pair's index.
    /// \param auxShell The auxiliary shell's index.
    /// \param pairFactor The canonical pair's ordered multiplicity.
    /// \returns True when the task is retained.
    bool Retains(std::size_t pairIndex, std::size_t auxShell, double pairFactor) const noexcept {
        if (!active)
        {
            return true;
        }

        const TwoElectronPairBound& bra = pairBounds[pairIndex];
        const TwoElectronPairBound& ket = auxDiagonalBounds[auxShell];

        if (!bra.hasBound() || !ket.hasBound())
        {
            return true;
        }

        // The energy path's predicate: Q_uv * Q_P, density-free.
        if (bra.value * ket.value >= threshold)
        {
            return true;
        }

        // The derivative-aware predicate: the task's bound contribution,
        // weighted by its density and the coefficients' shell maxima.
        const QuartetDerivativeBound bound = QuartetDerivativeProduct(bra, ket);
        const double scaled = pairFactor * pairDensity[pairIndex] * auxWeightMax[auxShell];
        return scaled * std::max(bound.value, bound.derivative) >= threshold;
    }
};

/// The four shells a three-index task's quartet is built from: the orbital
/// pair and the auxiliary shell with its phantom.
/// \param shells The flattened orbital shells.
/// \param pair The canonical orbital pair.
/// \param auxShells The flattened auxiliary shells.
/// \param phantoms The phantom shells, one per auxiliary shell.
/// \returns The tier's quartet.
inline MdEriDerivativeQuartet TaskQuartet(const std::vector<MdShellInput>& shells,
                                          const ShellPairIndex& pair,
                                          const std::vector<MdShellInput>& auxShells,
                                          const std::vector<MdShellInput>& phantoms,
                                          std::size_t auxShell) noexcept {
    MdEriDerivativeQuartet quartet;
    quartet.shells = {&shells[pair.i], &shells[pair.j], &auxShells[auxShell], &phantoms[auxShell]};
    return quartet;
}

/// Contracts one packed task value or derivative block with the density into
/// the auxiliary density: d_P += pairFactor * sum_uv D_uv block_uvP. The
/// phantom's slot is the constant, so the packed index carries one aux
/// function per element with the block's (u, v, P, s) layout.
/// \param block The packed block.
/// \param nI Shell-u function count.
/// \param nJ Shell-v function count.
/// \param nK The auxiliary shell's function count.
/// \param pairFactor The canonical pair's ordered multiplicity.
/// \param densityMatrix The density in the basis set's function order.
/// \param offsetI Shell u's first function.
/// \param offsetJ Shell v's first function.
/// \param auxOffset The auxiliary shell's first function.
/// \param auxiliaryDensity Out: d, one per auxiliary function.
inline void AccumulateTaskDensity(std::span<const double> block,
                                  std::size_t nI,
                                  std::size_t nJ,
                                  std::size_t nK,
                                  double pairFactor,
                                  const Eigen::MatrixXd& densityMatrix,
                                  std::size_t offsetI,
                                  std::size_t offsetJ,
                                  std::size_t auxOffset,
                                  std::span<double> auxiliaryDensity) {
    for (std::size_t fa = 0; fa < nI; ++fa)
    {
        for (std::size_t fb = 0; fb < nJ; ++fb)
        {
            const double densityElement = densityMatrix(static_cast<Eigen::Index>(offsetI + fa),
                                                        static_cast<Eigen::Index>(offsetJ + fb));

            for (std::size_t fp = 0; fp < nK; ++fp)
            {
                auxiliaryDensity[auxOffset + fp] +=
                    pairFactor * densityElement *
                    block[EriBlockIndex(fa, fb, fp, 0, nI, nJ, nK, 1)];
            }
        }
    }
}

/// Applies one differentiated metric block, M^(U) x, with both orientations
/// of an off-diagonal auxiliary pair. The metric is not screened: every
/// canonical pair is differentiated.
/// \param auxShells The flattened auxiliary shells.
/// \param auxPairList The auxiliary shell pair list.
/// \param phantoms The phantom shells, one per auxiliary shell.
/// \param tuple The differentiated tuple, in request order.
/// \param vector The vector the metric derivative multiplies.
/// \param out In-out: out += M^(tuple) vector.
/// \param scratch The tier's scratch.
/// \param nAtoms The geometry's atom count.
/// \returns Nothing, or the tier's refusal error.
inline qcx::Result<void> ApplyMetricDerivative(const std::vector<MdShellInput>& auxShells,
                                               const ShellPairList& auxPairList,
                                               const std::vector<MdShellInput>& phantoms,
                                               std::span<const std::size_t> tuple,
                                               const Eigen::VectorXd& vector,
                                               Eigen::VectorXd& out,
                                               MdEriDerivativeScratch& scratch,
                                               std::size_t nAtoms) {
    const int order = static_cast<int>(tuple.size());
    std::vector<double> block;

    for (const ShellPairIndex& pair : auxPairList.pairs)
    {
        const ShellInfo& braInfo = auxPairList.shells[pair.i];
        const ShellInfo& ketInfo = auxPairList.shells[pair.j];
        const std::size_t nI = ShellFunctionCount(braInfo);
        const std::size_t nJ = ShellFunctionCount(ketInfo);
        bool touches = true;

        for (const std::size_t coordinate : tuple)
        {
            const std::size_t atom = coordinate / 3;

            if (atom != braInfo.atomIndex && atom != ketInfo.atomIndex)
            {
                touches = false;
                break;
            }
        }

        if (!touches)
        {
            continue;
        }

        MdEriDerivativeQuartet quartet;
        quartet.shells = {
            &phantoms[pair.i], &auxShells[pair.i], &phantoms[pair.j], &auxShells[pair.j]};
        quartet.atoms = {
            braInfo.atomIndex, braInfo.atomIndex, ketInfo.atomIndex, ketInfo.atomIndex};
        block.assign(nI * nJ, 0.0);
        auto differentiated =
            BuildEriQuartetDerivative(quartet, order, tuple, block, scratch, nAtoms);

        if (!differentiated.has_value())
        {
            return std::unexpected(differentiated.error());
        }

        for (std::size_t fp = 0; fp < nI; ++fp)
        {
            for (std::size_t fq = 0; fq < nJ; ++fq)
            {
                const double value = block[EriBlockIndex(0, fp, 0, fq, 1, nI, 1, nJ)];
                out(static_cast<Eigen::Index>(braInfo.functionOffset + fp)) +=
                    value * vector(static_cast<Eigen::Index>(ketInfo.functionOffset + fq));

                // An off-diagonal canonical pair stands for both orientations
                // of the symmetric metric; a diagonal one's full square is
                // already the whole block.
                if (pair.i != pair.j)
                {
                    out(static_cast<Eigen::Index>(ketInfo.functionOffset + fq)) +=
                        value * vector(static_cast<Eigen::Index>(braInfo.functionOffset + fp));
                }
            }
        }
    }

    return {};
}

/// Accumulates one tuple's differentiated auxiliary density over the
/// retained tasks: d^(U)_P += sum_uv D_uv d(uv|P)^(U).
/// \param shells The flattened orbital shells.
/// \param pairList The orbital shell pair list.
/// \param auxShells The flattened auxiliary shells.
/// \param auxPairList The auxiliary shell pair list.
/// \param phantoms The phantom shells, one per auxiliary shell.
/// \param densityMatrix The density in the basis set's function order.
/// \param screen The retained task set.
/// \param tuple The differentiated tuple, in request order.
/// \param out Out: d^(U), one per auxiliary function.
/// \param scratch The tier's scratch.
/// \param nAtoms The geometry's atom count.
/// \returns Nothing, or the tier's refusal error.
inline qcx::Result<void> AccumulateAuxiliaryDensityDerivative(
    const std::vector<MdShellInput>& shells,
    const ShellPairList& pairList,
    const std::vector<MdShellInput>& auxShells,
    const ShellPairList& auxPairList,
    const std::vector<MdShellInput>& phantoms,
    const Eigen::MatrixXd& densityMatrix,
    const RiTaskScreen& screen,
    std::span<const std::size_t> tuple,
    Eigen::VectorXd& out,
    MdEriDerivativeScratch& scratch,
    std::size_t nAtoms) {
    const int order = static_cast<int>(tuple.size());
    std::vector<double> block;

    for (std::size_t pairIndex = 0; pairIndex < pairList.pairs.size(); ++pairIndex)
    {
        const ShellPairIndex& pair = pairList.pairs[pairIndex];
        const ShellInfo& braInfo = pairList.shells[pair.i];
        const ShellInfo& ketInfo = pairList.shells[pair.j];
        const std::size_t nI = ShellFunctionCount(braInfo);
        const std::size_t nJ = ShellFunctionCount(ketInfo);
        const double pairFactor = static_cast<double>(PairMultiplicity(pair));

        for (std::size_t auxShell = 0; auxShell < auxShells.size(); ++auxShell)
        {
            if (!screen.Retains(pairIndex, auxShell, pairFactor))
            {
                continue;
            }

            const ShellInfo& auxInfo = auxPairList.shells[auxShell];
            bool touches = true;

            for (const std::size_t coordinate : tuple)
            {
                const std::size_t atom = coordinate / 3;

                if (atom != braInfo.atomIndex && atom != ketInfo.atomIndex &&
                    atom != auxInfo.atomIndex)
                {
                    touches = false;
                    break;
                }
            }

            if (!touches)
            {
                continue;
            }

            MdEriDerivativeQuartet quartet =
                TaskQuartet(shells, pair, auxShells, phantoms, auxShell);
            quartet.atoms = {
                braInfo.atomIndex, ketInfo.atomIndex, auxInfo.atomIndex, auxInfo.atomIndex};
            const std::size_t elements = EriQuartetBlockElements(quartet);
            block.assign(elements, 0.0);
            auto differentiated =
                BuildEriQuartetDerivative(quartet, order, tuple, block, scratch, nAtoms);

            if (!differentiated.has_value())
            {
                return std::unexpected(differentiated.error());
            }

            const std::size_t nK = EriShellFunctions(*quartet.shells[2]);
            AccumulateTaskDensity(
                block,
                nI,
                nJ,
                nK,
                pairFactor,
                densityMatrix,
                braInfo.functionOffset,
                ketInfo.functionOffset,
                auxInfo.functionOffset,
                std::span<double>(out.data(), static_cast<std::size_t>(out.size())));
        }
    }

    return {};
}

} // namespace qcx::integrals::internal
