// The RI Coulomb gradient, at fixed density (ri_gradient.hpp). Two terms,
// both evaluated through the two-electron derivative tier on the quartet
// representation the energy path already evaluates the integrals in: the
// auxiliary shell paired with a phantom s function of exponent 0, so (uv|P)
// is the quartet (uv|P|s) and (P|Q) is (s|P|s|Q).
//
// The tier's primitive kernel forms its ket exponent as c + d, so the
// phantom's zero exponent never divides: q is the auxiliary shell's own
// exponent and the phantom folds into the pair's prefactor exactly as the
// energy path's BuildAuxPairData folds it. The phantom's own operators are
// the zero-exponent ones - a raise at 2 * 0 and a lower at index 0 - so the
// constant the phantom stands for contributes no derivative of its own,
// which is what a function that does not move should contribute.
//
// The screen is the never-under quartet bound: the orbital pair's own
// derivative-aware bound against the auxiliary shell's, times the pair
// block's largest density element. Both factors are maxima over the
// quantities they bound and the product rule bounds the quartet's
// derivative, so a discarded task's contribution is below the threshold at
// both orders - the screen is a derivative-aware one, not a value screen
// wearing its name.

#include "qcx/integrals/ri_gradient.hpp"

#include "internal/fock_screen.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_eri_derivative.hpp"
#include "internal/shells_flat.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace qcx::integrals {

namespace {

using qcx::integrals::internal::MdEriDerivativeQuartet;
using qcx::integrals::internal::MdEriDerivativeScratch;
using qcx::integrals::internal::MdShellInput;

/// The phantom s function an auxiliary shell is paired with: angular
/// momentum 0, exponent 0 and unit coefficient, at the auxiliary shell's own
/// centre. Exponent 0 makes the pair's second function the constant 1, so
/// the pair reproduces the bare Coulomb operator the resolution of identity
/// fits against - the same phantom the energy path's auxiliary pair data
/// carries.
/// \param center The auxiliary shell's centre (Bohr).
/// \returns The phantom shell input.
MdShellInput PhantomShell(const std::array<double, 3>& center) {
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

/// One assembled quartet: the derivative tier's view of it, the coordinates
/// its blocks are built for, and the value block.
struct GradientQuartet {
    MdEriDerivativeQuartet quartet{}; ///< The tier's shells and atoms.
    std::vector<std::size_t> coordinates; ///< Its own atoms' axes.
    std::vector<double> value; ///< The packed value block.
};

/// Fills one quartet's shells, atoms, coordinates and value block.
/// \param shells The four shells, in quartet order.
/// \param atoms The atom each shell sits on.
/// \param scratch The tier's scratch.
/// \param target The quartet to fill.
void FillGradientQuartet(const std::array<const MdShellInput*, 4>& shells,
                         const std::array<std::size_t, 4>& atoms,
                         MdEriDerivativeScratch& scratch,
                         GradientQuartet& target) {
    for (int slot = 0; slot < 4; ++slot)
    {
        const auto index = static_cast<std::size_t>(slot);
        target.quartet.shells[index] = shells[index];
        target.quartet.atoms[index] = atoms[index];
    }

    QuartetCoordinates(atoms, target.coordinates);
    target.value.assign(internal::EriQuartetBlockElements(target.quartet), 0.0);
    internal::BuildEriQuartetValue(target.quartet, 1, target.value, scratch);
}

/// The metric's floored eigen-inverse applied to one vector: the energy
/// path's solve, at the same floor, so the coefficients this gradient
/// differentiates are the coefficients the energy used.
/// \param metric The auxiliary metric, nAux x nAux, symmetric.
/// \param vector The right-hand side.
/// \param floorEpsilon The relative eigenvalue floor.
/// \returns M^-1 vector, or an Error (kInvalidArgument) when the metric is
/// degenerate below the floor.
qcx::Result<Eigen::VectorXd> MetricInverseSolve(const Eigen::MatrixXd& metric,
                                                const Eigen::VectorXd& vector,
                                                double floorEpsilon) {
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
    Eigen::VectorXd inverseEigenvalues(eigenvalues.size());

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

    return solver.eigenvectors() *
           inverseEigenvalues.cwiseProduct(solver.eigenvectors().transpose() * vector);
}

/// The largest magnitude the fitting coefficients carry over one auxiliary
/// shell's functions - the weight a screen has to bound a whole shell by.
/// \param w The coefficients, one per auxiliary function.
/// \param offset The shell's first auxiliary function.
/// \param nFuncs The shell's function count.
/// \returns The largest |w| over the shell.
double AuxiliaryWeightMax(const Eigen::VectorXd& w, std::size_t offset, std::size_t nFuncs) {
    double largest = 0.0;

    for (std::size_t f = 0; f < nFuncs; ++f)
    {
        largest = std::max(largest, std::abs(w(static_cast<Eigen::Index>(offset + f))));
    }

    return largest;
}

/// The basis set's shells, checked against the engine's angular-momentum cap.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \returns The shell pair list, or an Error.
qcx::Result<ShellPairList> CheckedShellPairs(const qcx::molecule::Molecule& molecule,
                                             const qcx::basisset::BasisSet& basisSet) {
    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    for (const ShellInfo& shell : pairList->shells)
    {
        if (!SupportsL(shell.angularMomentum))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "shell angular momentum exceeds kMaxEngineL of this build"});
        }
    }

    return pairList;
}

/// One canonical pair's never-under derivative-aware bound, and the pair
/// index it belongs to.
struct PairBoundTable {
    std::vector<TwoElectronPairBound> bounds; ///< One per canonical pair.
    std::vector<std::size_t> diagonal; ///< The (a, a) canonical pair of each shell.
};

/// Builds a basis set's pair bounds and its per-shell diagonal lookup.
/// \param molecule The molecule.
/// \param basisSet The basis set.
/// \param pairList Its shell pair list.
/// \returns The table, or an Error.
qcx::Result<PairBoundTable> BuildPairBoundTable(const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& basisSet,
                                                const ShellPairList& pairList) {
    auto bounds = ComputeTwoElectronPairBounds(molecule, basisSet);

    if (!bounds.has_value())
    {
        return std::unexpected(bounds.error());
    }

    PairBoundTable table;
    table.bounds = std::move(*bounds);
    table.diagonal.resize(pairList.shells.size());

    for (std::size_t shell = 0; shell < pairList.shells.size(); ++shell)
    {
        table.diagonal[shell] = PairIndexOf(shell, shell, pairList);
    }

    return table;
}

/// Accumulates one differentiated quartet into the gradient: the block's
/// elements weighted by the caller's coefficients and added to the entries
/// of the coordinates' atoms.
/// \param built The quartet, its coordinates and its value block.
/// \param derivatives The derivative blocks, one packed block per coordinate.
/// \param weights The coefficient that multiplies each block element.
/// \param gradient Out: the 3N gradient.
void AccumulateQuartetGradient(const GradientQuartet& built,
                               std::span<const double> derivatives,
                               std::span<const double> weights,
                               std::span<double> gradient) {
    const std::size_t elements = built.value.size();

    for (std::size_t coordinate = 0; coordinate < built.coordinates.size(); ++coordinate)
    {
        const std::span<const double> block = derivatives.subspan(coordinate * elements, elements);
        double sum = 0.0;

        for (std::size_t element = 0; element < elements; ++element)
        {
            sum += weights[element] * block[element];
        }

        gradient[built.coordinates[coordinate]] += sum;
    }
}

} // namespace

qcx::Result<RiCoulombGradient> ComputeRiCoulombGradient(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::basisset::BasisSet& auxBasisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    const RiEngineOptions& options) {
    auto pairList = CheckedShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto auxPairList = CheckedShellPairs(molecule, auxBasisSet);

    if (!auxPairList.has_value())
    {
        return std::unexpected(auxPairList.error());
    }

    const std::size_t n = pairList->functionCount;
    const std::size_t nAux = auxPairList->functionCount;

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

    auto auxShells = internal::FlattenShells(molecule, auxBasisSet, *auxPairList);

    if (!auxShells.has_value())
    {
        return std::unexpected(auxShells.error());
    }

    // The energy path's own metric and three-index tensor: the gradient
    // differentiates the coefficients these produce, not a second solve.
    auto metric = BuildAuxMetric(molecule, auxBasisSet, options);

    if (!metric.has_value())
    {
        return std::unexpected(metric.error());
    }

    auto riTensor = BuildRiTensor(molecule, basisSet, auxBasisSet, options);

    if (!riTensor.has_value())
    {
        return std::unexpected(riTensor.error());
    }

    const std::size_t nAtoms = molecule.AtomCount();
    RiCoulombGradient result;
    result.gradient = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(3 * nAtoms));

    if (n == 0 || nAux == 0)
    {
        return result;
    }

    Eigen::MatrixXd metricMatrix(static_cast<Eigen::Index>(nAux), static_cast<Eigen::Index>(nAux));

    for (std::size_t p = 0; p < nAux; ++p)
    {
        for (std::size_t q = 0; q < nAux; ++q)
        {
            metricMatrix(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q)) =
                (*metric)(p, q);
        }
    }

    // d_P = sum_uv D_uv (uv|P): the density in the auxiliary space, read off
    // the tensor the energy path builds.
    Eigen::VectorXd auxiliaryDensity = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nAux));

    for (std::size_t p = 0; p < nAux; ++p)
    {
        double sum = 0.0;

        for (std::size_t u = 0; u < n; ++u)
        {
            for (std::size_t v = 0; v < n; ++v)
            {
                sum += density(u, v) * (*riTensor)(u, v, p);
            }
        }

        auxiliaryDensity(static_cast<Eigen::Index>(p)) = sum;
    }

    // w = M^-1 d, the fitting coefficients the energy path's contraction
    // multiplies the tensor by.
    auto coefficients =
        MetricInverseSolve(metricMatrix, auxiliaryDensity, options.metricFloorEpsilon);

    if (!coefficients.has_value())
    {
        return std::unexpected(coefficients.error());
    }

    const Eigen::VectorXd& w = *coefficients;
    result.energy = 0.5 * auxiliaryDensity.dot(w);

    // The screen: the orbital pair's own derivative-aware bound against the
    // auxiliary shell's, times the largest density element of the pair's
    // block. Every factor is a maximum over what it bounds and the product
    // rule bounds the quartet's derivative, so the discarded contribution is
    // below the threshold at both orders. The slack covers the bound's own
    // floating point, which is the only place it can be under.
    const double threshold =
        SchwarzThreshold(options.accuracy) / (1.0 + internal::kNeighborListSlack);
    auto pairBounds = BuildPairBoundTable(molecule, basisSet, *pairList);

    if (!pairBounds.has_value())
    {
        return std::unexpected(pairBounds.error());
    }

    auto auxPairBounds = BuildPairBoundTable(molecule, auxBasisSet, *auxPairList);

    if (!auxPairBounds.has_value())
    {
        return std::unexpected(auxPairBounds.error());
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

    const std::vector<double> pairDensity =
        internal::BuildShellPairMaxDensity(densityMatrix, *pairList);

    // The phantom each auxiliary shell is paired with, one per auxiliary
    // shell, at that shell's own centre: a function that does not move, so
    // the tier's operator set for the auxiliary atom is the auxiliary
    // shell's own.
    std::vector<MdShellInput> phantoms;
    phantoms.reserve(auxShells->size());

    for (const MdShellInput& shell : *auxShells)
    {
        phantoms.push_back(PhantomShell({shell.cx, shell.cy, shell.cz}));
    }

    MdEriDerivativeScratch scratch;
    GradientQuartet built;
    std::vector<double> derivatives;
    std::vector<double> weights;

    // The tensor term: sum_P w_P d(d_P)/dR, one differentiated quartet per
    // canonical orbital pair and auxiliary shell.
    for (const ShellPairIndex& pair : pairList->pairs)
    {
        const ShellInfo& braInfo = pairList->shells[pair.i];
        const ShellInfo& ketInfo = pairList->shells[pair.j];
        const std::size_t nI = ShellFunctionCount(braInfo);
        const std::size_t nJ = ShellFunctionCount(ketInfo);
        // A canonical pair enumerates each unordered pair once; the ordered
        // sum the energy takes counts an off-diagonal pair twice.
        const double pairFactor = pair.i == pair.j ? 1.0 : 2.0;
        const std::size_t pairIndex = PairIndexOf(pair.i, pair.j, *pairList);
        const TwoElectronPairBound& braBound = pairBounds->bounds[pairIndex];
        const double densityWeight = pairDensity[pairIndex];
        ++result.counts.orbitalPairs;

        for (std::size_t auxShell = 0; auxShell < auxShells->size(); ++auxShell)
        {
            const ShellInfo& auxInfo = auxPairList->shells[auxShell];
            const TwoElectronPairBound& ketBound =
                auxPairBounds->bounds[auxPairBounds->diagonal[auxShell]];
            const QuartetDerivativeBound bound = QuartetDerivativeProduct(braBound, ketBound);
            const double scaled =
                pairFactor * densityWeight *
                AuxiliaryWeightMax(w, auxInfo.functionOffset, ShellFunctionCount(auxInfo));

            // The screen drops a task whose bound contribution is below the
            // threshold at both orders: the value bound alone would drop a
            // pair whose Schwarz bound passes through zero while its quartet
            // contributions do not.
            if (scaled * std::max(bound.value, bound.derivative) < threshold)
            {
                continue;
            }

            FillGradientQuartet(
                {&(*shells)[pair.i],
                 &(*shells)[pair.j],
                 auxShells->data() + auxShell,
                 phantoms.data() + auxShell},
                {braInfo.atomIndex, ketInfo.atomIndex, auxInfo.atomIndex, auxInfo.atomIndex},
                scratch,
                built);

            const std::size_t elements = built.value.size();
            derivatives.assign(built.coordinates.size() * elements, 0.0);
            auto differentiated = internal::BuildEriQuartetDerivative(
                built.quartet, 1, built.coordinates, derivatives, scratch, nAtoms);

            if (!differentiated.has_value())
            {
                return std::unexpected(differentiated.error());
            }

            const std::size_t nK = internal::EriShellFunctions(*built.quartet.shells[2]);
            weights.assign(elements, 0.0);

            for (std::size_t fa = 0; fa < nI; ++fa)
            {
                for (std::size_t fb = 0; fb < nJ; ++fb)
                {
                    const double densityElement =
                        density(braInfo.functionOffset + fa, ketInfo.functionOffset + fb);

                    for (std::size_t fp = 0; fp < nK; ++fp)
                    {
                        const double wP = w(static_cast<Eigen::Index>(auxInfo.functionOffset + fp));
                        weights[EriBlockIndex(fa, fb, fp, 0, nI, nJ, nK, 1)] =
                            pairFactor * densityElement * wP;
                    }
                }
            }

            AccumulateQuartetGradient(
                built,
                derivatives,
                weights,
                std::span<double>(result.gradient.data(),
                                  static_cast<std::size_t>(result.gradient.size())));
            ++result.counts.tensorQuartets;
        }
    }

    result.counts.auxShells = auxShells->size();

    // The metric term: -1/2 sum_PQ w_P w_Q d(M_PQ)/dR. The auxiliary metric
    // is the two-centre (P|Q), a quartet of two phantom-paired auxiliary
    // shells.
    for (const ShellPairIndex& pair : auxPairList->pairs)
    {
        const ShellInfo& braInfo = auxPairList->shells[pair.i];
        const ShellInfo& ketInfo = auxPairList->shells[pair.j];
        const std::size_t nI = ShellFunctionCount(braInfo);
        const std::size_t nJ = ShellFunctionCount(ketInfo);
        const double pairFactor = pair.i == pair.j ? 1.0 : 2.0;
        // The quartet is (s_i, P_i, s_j, Q_j): its bra pair is the shell i
        // diagonal pair and its ket pair the shell j one, so each factor of
        // the product rule takes its OWN diagonal pair's bound. The canonical
        // pair's own bound is neither of them - it bounds the (ij|ij) block,
        // which over the diagonal pair is not what the factor multiplies.
        const TwoElectronPairBound& braBound =
            auxPairBounds->bounds[auxPairBounds->diagonal[pair.i]];
        const TwoElectronPairBound& ketBound =
            auxPairBounds->bounds[auxPairBounds->diagonal[pair.j]];
        // The metric term is a fixed -1/2 sum over ordered pairs of w_P w_Q
        // times the pair's own derivative: its bound is the coefficients'
        // product against the pair's value and derivative bounds, through the
        // same product rule the tensor loop reads (a pair with no bound makes
        // the product unbounded rather than a NaN).
        const QuartetDerivativeBound bound = QuartetDerivativeProduct(braBound, ketBound);
        const double coefficientWeight = AuxiliaryWeightMax(w, braInfo.functionOffset, nI) *
                                         AuxiliaryWeightMax(w, ketInfo.functionOffset, nJ);

        if (pairFactor * coefficientWeight * std::max(bound.value, bound.derivative) < threshold)
        {
            continue;
        }

        FillGradientQuartet(
            {phantoms.data() + pair.i,
             auxShells->data() + pair.i,
             phantoms.data() + pair.j,
             auxShells->data() + pair.j},
            {braInfo.atomIndex, braInfo.atomIndex, ketInfo.atomIndex, ketInfo.atomIndex},
            scratch,
            built);

        const std::size_t elements = built.value.size();
        derivatives.assign(built.coordinates.size() * elements, 0.0);
        auto differentiated = internal::BuildEriQuartetDerivative(
            built.quartet, 1, built.coordinates, derivatives, scratch, nAtoms);

        if (!differentiated.has_value())
        {
            return std::unexpected(differentiated.error());
        }

        // The metric quartet's block is (1 x nI x 1 x nJ): the phantom slots
        // carry one function each, so the packed offset is that block's
        // (fP, fQ) element in the engine's pair-major layout.
        weights.assign(elements, 0.0);

        for (std::size_t fp = 0; fp < nI; ++fp)
        {
            for (std::size_t fq = 0; fq < nJ; ++fq)
            {
                const double wP = w(static_cast<Eigen::Index>(braInfo.functionOffset + fp));
                const double wQ = w(static_cast<Eigen::Index>(ketInfo.functionOffset + fq));
                weights[EriBlockIndex(0, fp, 0, fq, 1, nI, 1, nJ)] = -0.5 * pairFactor * wP * wQ;
            }
        }

        AccumulateQuartetGradient(
            built,
            derivatives,
            weights,
            std::span<double>(result.gradient.data(),
                              static_cast<std::size_t>(result.gradient.size())));
        ++result.counts.metricQuartets;
    }

    return result;
}

} // namespace qcx::integrals
