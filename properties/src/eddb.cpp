// EDDB via the Bond-Orbital Projection algorithm:
// the implementation-level formulation of [Szczepanik2017BOP] in the
// Gopinathan-Jug working basis (P_ort = S^{1/2} P S^{1/2}, populations.hpp's
// forward root). Two- and three-center bond-order-orbital eigenproblems
// over the density's interatomic blocks, phase-sensitive projections,
// upper-envelope selection, and the occupied-space congruence projection.
#include "qcx/properties/eddb.hpp"

#include "internal/tensor_to_eigen.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/properties/populations.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace qcx::properties {
namespace {

// The per-spin orthogonalized density in the working basis.
Eigen::MatrixXd OrthogonalizedDensity(const Eigen::MatrixXd& root, const Eigen::MatrixXd& density) {
    return root * density * root;
}

// The Wiberg-type bond orders of the orthogonalized densities (eq. 4-5 of
// [Szczepanik2017BOP] in the per-spin convention): W_AB = sum_spins sum_{mu in
// A, nu in B} (P^spin_ort)_mu,nu^2. The closed-shell factor 1/2 of the
// D-based definition is absorbed by the per-spin densities (P = D/2).
Eigen::MatrixXd WibergBondOrders(const Eigen::MatrixXd& alpha,
                                 const Eigen::MatrixXd& beta,
                                 const std::vector<AoRange>& aoRanges) {
    const std::size_t nAtoms = aoRanges.size();
    Eigen::MatrixXd wiberg =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nAtoms), static_cast<Eigen::Index>(nAtoms));

    for (std::size_t a = 0; a < nAtoms; ++a)
    {
        for (std::size_t b = a + 1; b < nAtoms; ++b)
        {
            double value = 0.0;
            const std::size_t firstA = aoRanges[a].firstFunction;
            const std::size_t firstB = aoRanges[b].firstFunction;

            for (std::size_t mu = 0; mu < aoRanges[a].functionCount; ++mu)
            {
                for (std::size_t nu = 0; nu < aoRanges[b].functionCount; ++nu)
                {
                    const Eigen::Index i = static_cast<Eigen::Index>(firstA + mu);
                    const Eigen::Index j = static_cast<Eigen::Index>(firstB + nu);
                    value += alpha(i, j) * alpha(i, j) + beta(i, j) * beta(i, j);
                }
            }

            wiberg(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) = value;
            wiberg(static_cast<Eigen::Index>(b), static_cast<Eigen::Index>(a)) = value;
        }
    }

    return wiberg;
}

// The global basis-function indices of one atom.
std::vector<Eigen::Index> AtomFunctions(const AoRange& range) {
    std::vector<Eigen::Index> indices;
    indices.reserve(range.functionCount);

    for (std::size_t mu = 0; mu < range.functionCount; ++mu)
    {
        indices.push_back(static_cast<Eigen::Index>(range.firstFunction + mu));
    }

    return indices;
}

// The two-center coupling P^(2)_AX = [[0, P_AX], [P_XA, 0]] in the local
// A + X ordering, its eigen-decomposition, and the retained bonding and
// antibonding bond-order orbitals re-expressed in the full working basis.
struct TwoCenterOrbitals {
    std::vector<Eigen::Index> atomA; ///< Global indices of A's functions.
    std::vector<Eigen::Index> atomX; ///< Global indices of X's functions.
    Eigen::MatrixXd bonding; ///< n x nAX columns, full-basis 2cBOOs.
    Eigen::MatrixXd antibonding; ///< n x nAX columns, paired antibonding 2cBOOs.
    Eigen::VectorXd occupations; ///< eta = lambda^2, ascending index = descending eigenvalue.
};

TwoCenterOrbitals BuildTwoCenterOrbitals(const Eigen::MatrixXd& density,
                                         const std::vector<Eigen::Index>& functionsA,
                                         const std::vector<Eigen::Index>& functionsX) {
    const Eigen::Index nA = static_cast<Eigen::Index>(functionsA.size());
    const Eigen::Index nX = static_cast<Eigen::Index>(functionsX.size());
    const Eigen::Index size = nA + nX;

    // The local coupling matrix with the atoms ordered [A | X].
    Eigen::MatrixXd coupling = Eigen::MatrixXd::Zero(size, size);

    for (Eigen::Index i = 0; i < nA; ++i)
    {
        for (Eigen::Index j = 0; j < nX; ++j)
        {
            coupling(i, nA + j) = density(functionsA[static_cast<std::size_t>(i)],
                                          functionsX[static_cast<std::size_t>(j)]);
            coupling(nA + j, i) = coupling(i, nA + j);
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(coupling);

    if (solver.info() != Eigen::Success)
    {
        return TwoCenterOrbitals{};
    }

    // Eigenvalues ascending; the +/- pairs of the off-diagonal structure
    // (rank = rank of the A-X block) give the bonding (positive, largest
    // first) and antibonding (negative, most negative first) members, at
    // most nAX = min(nA, nX) each.
    const Eigen::Index rank = std::min(nA, nX);
    const Eigen::VectorXd& values = solver.eigenvalues();
    const Eigen::MatrixXd& vectors = solver.eigenvectors();

    TwoCenterOrbitals result;
    result.atomA = functionsA;
    result.atomX = functionsX;
    const Eigen::Index n = static_cast<Eigen::Index>(density.rows());
    result.bonding = Eigen::MatrixXd::Zero(n, rank);
    result.antibonding = Eigen::MatrixXd::Zero(n, rank);
    result.occupations = Eigen::VectorXd::Zero(rank);

    Eigen::Index bonded = 0;
    Eigen::Index antibonded = 0;

    for (Eigen::Index k = size - 1; k >= 0 && bonded < rank; --k)
    {
        if (values(k) > 0.0)
        {
            // Local vector -> full basis (A part in rows 0..nA, X part in
            // rows nA..nA+nX).
            for (Eigen::Index i = 0; i < nA; ++i)
            {
                result.bonding(functionsA[static_cast<std::size_t>(i)], bonded) = vectors(i, k);
            }

            for (Eigen::Index j = 0; j < nX; ++j)
            {
                result.bonding(functionsX[static_cast<std::size_t>(j)], bonded) =
                    vectors(nA + j, k);
            }

            result.occupations(bonded) = values(k) * values(k);
            ++bonded;
        }
    }

    for (Eigen::Index k = 0; k < size && antibonded < rank; ++k)
    {
        if (values(k) < 0.0)
        {
            for (Eigen::Index i = 0; i < nA; ++i)
            {
                result.antibonding(functionsA[static_cast<std::size_t>(i)], antibonded) =
                    vectors(i, k);
            }

            for (Eigen::Index j = 0; j < nX; ++j)
            {
                result.antibonding(functionsX[static_cast<std::size_t>(j)], antibonded) =
                    vectors(nA + j, k);
            }

            ++antibonded;
        }
    }

    // A rank-deficient A-X block retains fewer than n_AX members; drop the
    // zero columns so the channel bookkeeping stays honest.
    result.bonding.conservativeResize(Eigen::NoChange, bonded);
    result.antibonding.conservativeResize(Eigen::NoChange, antibonded);
    result.occupations.conservativeResize(bonded);
    return result;
}

// The three-center through-bridge coupling P^(3)_AXB (eq. 12): the A-B
// block deliberately zero, rows ordered [A | X | B]. Returns the leading
// (at most n_X) bonding 3cBOOs as full-basis columns and their occupations.
struct ThreeCenterOrbitals {
    Eigen::MatrixXd bonding; ///< n x n3 columns, full-basis 3cBOOs.
    Eigen::VectorXd occupations; ///< theta = lambda^2.
};

ThreeCenterOrbitals BuildThreeCenterOrbitals(const Eigen::MatrixXd& density,
                                             const std::vector<Eigen::Index>& functionsA,
                                             const std::vector<Eigen::Index>& functionsX,
                                             const std::vector<Eigen::Index>& functionsB) {
    const Eigen::Index nA = static_cast<Eigen::Index>(functionsA.size());
    const Eigen::Index nX = static_cast<Eigen::Index>(functionsX.size());
    const Eigen::Index nB = static_cast<Eigen::Index>(functionsB.size());
    const Eigen::Index size = nA + nX + nB;

    Eigen::MatrixXd coupling = Eigen::MatrixXd::Zero(size, size);

    for (Eigen::Index i = 0; i < nA; ++i)
    {
        for (Eigen::Index j = 0; j < nX; ++j)
        {
            coupling(i, nA + j) = density(functionsA[static_cast<std::size_t>(i)],
                                          functionsX[static_cast<std::size_t>(j)]);
            coupling(nA + j, i) = coupling(i, nA + j);
        }
    }

    for (Eigen::Index j = 0; j < nX; ++j)
    {
        for (Eigen::Index k = 0; k < nB; ++k)
        {
            coupling(nA + j, nA + nX + k) = density(functionsX[static_cast<std::size_t>(j)],
                                                    functionsB[static_cast<std::size_t>(k)]);
            coupling(nA + nX + k, nA + j) = coupling(nA + j, nA + nX + k);
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(coupling);

    ThreeCenterOrbitals result;
    result.bonding =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(density.rows()), std::min(nX, size));
    result.occupations = Eigen::VectorXd::Zero(std::min(nX, size));

    if (solver.info() != Eigen::Success)
    {
        return result;
    }

    // The leading n_X bonding (positive) eigenvalues, descending.
    const Eigen::VectorXd& values = solver.eigenvalues();
    const Eigen::MatrixXd& vectors = solver.eigenvectors();
    Eigen::Index retained = 0;

    for (Eigen::Index k = size - 1; k >= 0 && retained < nX; --k)
    {
        if (values(k) <= 0.0)
        {
            break;
        }

        for (Eigen::Index i = 0; i < nA; ++i)
        {
            result.bonding(functionsA[static_cast<std::size_t>(i)], retained) = vectors(i, k);
        }

        for (Eigen::Index j = 0; j < nX; ++j)
        {
            result.bonding(functionsX[static_cast<std::size_t>(j)], retained) = vectors(nA + j, k);
        }

        for (Eigen::Index l = 0; l < nB; ++l)
        {
            result.bonding(functionsB[static_cast<std::size_t>(l)], retained) =
                vectors(nA + nX + l, k);
        }

        result.occupations(retained) = values(k) * values(k);
        ++retained;
    }

    result.bonding.conservativeResize(Eigen::NoChange, retained);
    result.occupations.conservativeResize(retained);
    return result;
}

// The occupation-dependent scaling g_tau of eq. (17).
double ScaleOccupation(double occupation, double threshold) {
    return occupation >= threshold ? occupation / threshold : 1.0;
}

// The thresholded inverse square root: directions with eigenvalue below
// the threshold are removed (the "tau" subscript of eq. 18). The cutoff
// never falls below a relative epsilon floor so the projection stays
// well-defined for a (numerically) singular Gram matrix.
Eigen::MatrixXd ThresholdedInverseRoot(const Eigen::MatrixXd& gram, double threshold) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(gram);

    if (solver.info() != Eigen::Success)
    {
        return Eigen::MatrixXd::Zero(gram.rows(), gram.cols());
    }

    const Eigen::VectorXd& values = solver.eigenvalues();
    const Eigen::MatrixXd& vectors = solver.eigenvectors();
    const double scale = values.size() > 0 ? values(values.size() - 1) : 0.0;
    const double cutoff = std::max(threshold, scale * std::numeric_limits<double>::epsilon() * 8.0);
    Eigen::MatrixXd inverse = Eigen::MatrixXd::Zero(gram.rows(), gram.cols());

    for (Eigen::Index k = 0; k < values.size(); ++k)
    {
        if (values(k) > cutoff)
        {
            inverse += vectors.col(k) * (vectors.col(k).transpose() / std::sqrt(values(k)));
        }
    }

    return inverse;
}

// The raw BOP layer K of one central atom (eq. 29-31): the bonding and
// antibonding 2cBOOs of all its selected neighbors, weighted by the
// retained phase-coherence populations x.
Eigen::MatrixXd CentralAtomLayer(const std::vector<TwoCenterOrbitals>& orbitalSets,
                                 const std::vector<Eigen::VectorXd>& retainedPopulations,
                                 const Eigen::Index n) {
    Eigen::MatrixXd layer = Eigen::MatrixXd::Zero(n, n);

    for (std::size_t s = 0; s < orbitalSets.size(); ++s)
    {
        const Eigen::MatrixXd& bonding = orbitalSets[s].bonding;
        const Eigen::MatrixXd& antibonding = orbitalSets[s].antibonding;
        const Eigen::VectorXd& x = retainedPopulations[s];
        layer.noalias() += bonding * x.asDiagonal() * bonding.transpose();
        layer.noalias() += antibonding * x.asDiagonal() * antibonding.transpose();
    }

    return layer;
}

// The per-spin raw BOP layer K over the central-atom loop (eq. 29-31):
// every central atom's bonding and antibonding 2cBOOs weighted by the
// retained phase-coherence populations x.  Returns the layer and the
// total number of retained two-center bond-order orbitals.
std::pair<Eigen::MatrixXd, std::size_t> RawBopLayer(
    const Eigen::MatrixXd& density,
    const std::vector<std::vector<std::size_t>>& neighbors,
    const std::vector<AoRange>& aoRanges,
    double tauBop) {
    const Eigen::Index n = density.rows();
    const std::size_t nAtoms = neighbors.size();
    Eigen::MatrixXd k = Eigen::MatrixXd::Zero(n, n);
    std::size_t orbitalCount = 0;

    for (std::size_t x = 0; x < nAtoms; ++x)
    {
        if (neighbors[x].size() < 2)
        {
            continue;
        }

        // The 2cBOOs of every selected neighbor (eq. 8-11).
        std::vector<TwoCenterOrbitals> orbitalSets;
        orbitalSets.reserve(neighbors[x].size());

        for (const std::size_t a : neighbors[x])
        {
            orbitalSets.push_back(BuildTwoCenterOrbitals(
                density, AtomFunctions(aoRanges[a]), AtomFunctions(aoRanges[x])));
            orbitalCount += orbitalSets.back().occupations.size();
        }

        // The retained phase-coherence populations x (eq. 25-28):
        // each 2cBOO keeps the largest w over its triplets, capped by
        // its own occupation.
        std::vector<Eigen::VectorXd> retained;
        retained.reserve(orbitalSets.size());

        for (const TwoCenterOrbitals& orbitals : orbitalSets)
        {
            retained.push_back(Eigen::VectorXd::Zero(orbitals.occupations.size()));
        }

        for (std::size_t t = 0; t < orbitalSets.size(); ++t)
        {
            for (std::size_t u = t + 1; u < orbitalSets.size(); ++u)
            {
                // The triplet A_t - X - A_u (A_t and A_u the two
                // neighbors, A < B in the neighbor list).
                const TwoCenterOrbitals& first = orbitalSets[t];
                const TwoCenterOrbitals& second = orbitalSets[u];
                ThreeCenterOrbitals triple =
                    BuildThreeCenterOrbitals(density, first.atomA, first.atomX, second.atomA);

                if (triple.occupations.size() == 0)
                {
                    continue;
                }

                // The projection matrix (eq. 16): the 3cBOOs expanded
                // in the combined two-center bond-orbital space of the
                // two adjacent bonds, with the occupation scaling.
                const Eigen::Index nChannels = first.bonding.cols() + second.bonding.cols();
                Eigen::MatrixXd projection =
                    Eigen::MatrixXd::Zero(nChannels, triple.bonding.cols());

                projection.topRows(first.bonding.cols()).noalias() =
                    first.bonding.transpose() * triple.bonding;
                projection.bottomRows(second.bonding.cols()).noalias() =
                    second.bonding.transpose() * triple.bonding;

                for (Eigen::Index i = 0; i < nChannels; ++i)
                {
                    projection.row(i) *= ScaleOccupation(
                        i < first.bonding.cols() ? first.occupations(i)
                                                 : second.occupations(i - first.bonding.cols()),
                        tauBop);
                }

                for (Eigen::Index k = 0; k < triple.bonding.cols(); ++k)
                {
                    projection.col(k) *= ScaleOccupation(triple.occupations(k), tauBop);
                }

                // The thresholded Lowdin orthogonalization (eq. 18).
                const Eigen::MatrixXd gram = projection.transpose() * projection;
                Eigen::MatrixXd orthogonal = projection * ThresholdedInverseRoot(gram, tauBop);

                // The signed cumulative coherence (eq. 19-22): channels
                // accumulate BEFORE squaring so conjugating and
                // anticonjugating combinations cancel.
                const Eigen::Index nFirst = first.bonding.cols();
                Eigen::VectorXd channelPopulations = Eigen::VectorXd::Zero(orthogonal.cols());
                Eigen::MatrixXd cumulative = Eigen::MatrixXd::Zero(nFirst, second.bonding.cols());
                double previousNorm = 0.0;

                for (Eigen::Index k = 0; k < orthogonal.cols(); ++k)
                {
                    const Eigen::VectorXd cFirst = orthogonal.col(k).head(nFirst);
                    const Eigen::VectorXd cSecond = orthogonal.col(k).tail(second.bonding.cols());
                    cumulative.noalias() += cFirst * cSecond.transpose();
                    const double coherence = 4.0 * cumulative.squaredNorm();
                    channelPopulations(k) = (coherence - previousNorm) * triple.occupations(k);
                    previousNorm = coherence;
                }

                // The residual negative populations from the
                // orthogonalization are redistributed to the most
                // closely related positive channel (eq. 23-24).
                for (Eigen::Index i = 0; i < channelPopulations.size(); ++i)
                {
                    if (channelPopulations(i) >= 0.0)
                    {
                        continue;
                    }

                    Eigen::Index best = -1;
                    double bestOverlap = 0.0;

                    for (Eigen::Index j = 0; j < channelPopulations.size(); ++j)
                    {
                        if (j == i || channelPopulations(j) <= 0.0)
                        {
                            continue;
                        }

                        const double firstOverlap = std::abs(
                            orthogonal.col(i).head(nFirst).dot(orthogonal.col(j).head(nFirst)));
                        const double secondOverlap =
                            std::abs(orthogonal.col(i)
                                         .tail(second.bonding.cols())
                                         .dot(orthogonal.col(j).tail(second.bonding.cols())));

                        if (firstOverlap * secondOverlap > bestOverlap)
                        {
                            bestOverlap = firstOverlap * secondOverlap;
                            best = j;
                        }
                    }

                    if (best >= 0)
                    {
                        channelPopulations(best) += channelPopulations(i);
                    }

                    channelPopulations(i) = 0.0;
                }

                // The candidate w for every participating 2cBOO of both
                // adjacent bonds (eq. 25): the corrected coherence
                // distributed over the squared projection coefficients.
                for (Eigen::Index i = 0; i < nChannels; ++i)
                {
                    double w = 0.0;

                    for (Eigen::Index k = 0; k < orthogonal.cols(); ++k)
                    {
                        w += channelPopulations(k) * orthogonal(i, k) * orthogonal(i, k);
                    }

                    Eigen::VectorXd& target = i < nFirst ? retained[t] : retained[u];
                    const Eigen::Index local = i < nFirst ? i : i - nFirst;
                    const double occupation =
                        i < nFirst ? first.occupations(local) : second.occupations(local);
                    target(local) = std::min(occupation, std::max(target(local), w));
                }
            }
        }

        // The central atom's contribution to the raw layer (eq. 29-31):
        // bonding and antibonding 2cBOOs share the retained x.
        k.noalias() += CentralAtomLayer(orbitalSets, retained, n);
    }

    return std::pair<Eigen::MatrixXd, std::size_t>{std::move(k), orbitalCount};
}

} // namespace

qcx::Result<EddbAnalysis> AnalyzeEddb(const qcx::molecule::Molecule& molecule,
                                      const qcx::basisset::BasisSet& basisSet,
                                      const Eigen::MatrixXd& densityAlpha,
                                      const Eigen::MatrixXd& densityBeta,
                                      const EddbOptions& options) {
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    const std::size_t n = overlap->Shape()[0];

    if (densityAlpha.rows() != static_cast<Eigen::Index>(n) ||
        densityAlpha.cols() != static_cast<Eigen::Index>(n) ||
        densityBeta.rows() != static_cast<Eigen::Index>(n) ||
        densityBeta.cols() != static_cast<Eigen::Index>(n))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "densities must be n x n with n the AO count"});
    }

    auto aoRanges = AoIndexRangesByAtom(molecule, basisSet);

    if (!aoRanges.has_value())
    {
        return std::unexpected(aoRanges.error());
    }

    const Eigen::MatrixXd root = SymmetricSquareRootOfOverlap(internal::TensorToEigen(*overlap));
    const Eigen::MatrixXd alpha = OrthogonalizedDensity(root, densityAlpha);
    const Eigen::MatrixXd beta = OrthogonalizedDensity(root, densityBeta);

    // The connectivity mask from the Wiberg-type bond orders (eq. 4-6).
    const Eigen::MatrixXd wiberg = WibergBondOrders(alpha, beta, *aoRanges);
    const std::size_t nAtoms = aoRanges->size();
    std::vector<std::vector<std::size_t>> neighbors(nAtoms);

    for (std::size_t a = 0; a < nAtoms; ++a)
    {
        for (std::size_t b = a + 1; b < nAtoms; ++b)
        {
            if (wiberg(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) >
                options.wibergThreshold)
            {
                neighbors[a].push_back(b);
                neighbors[b].push_back(a);
            }
        }
    }

    auto [rawAlpha, countAlpha] = RawBopLayer(alpha, neighbors, *aoRanges, options.bopThreshold);
    auto [rawBeta, countBeta] = RawBopLayer(beta, neighbors, *aoRanges, options.bopThreshold);

    // The occupied-space congruence projection (eq. 32-33): the raw layer
    // may carry virtual-space components; D K D (per spin) removes them.
    const Eigen::MatrixXd delocalized = 2.0 * (alpha * rawAlpha * alpha + beta * rawBeta * beta);

    EddbAnalysis result;
    result.delocalizedDensity = delocalized;
    result.totalPopulation = delocalized.trace();
    result.atomicPopulations = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nAtoms));

    for (std::size_t a = 0; a < nAtoms; ++a)
    {
        double population = 0.0;

        for (std::size_t mu = 0; mu < (*aoRanges)[a].functionCount; ++mu)
        {
            const Eigen::Index i = static_cast<Eigen::Index>((*aoRanges)[a].firstFunction + mu);
            population += delocalized(i, i);
        }

        result.atomicPopulations(static_cast<Eigen::Index>(a)) = population;
    }

    // The NOBD resolution (eq. 38): occupations of the delocalization
    // natural orbitals, descending.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(delocalized);
    result.nobdOccupations = solver.eigenvalues().reverse();

    result.centralAtomCount = static_cast<std::size_t>(std::count_if(
        neighbors.begin(), neighbors.end(), [](const auto& list) { return list.size() >= 2; }));
    result.twoCenterOrbitalCount = std::max(countAlpha, countBeta);
    return result;
}

} // namespace qcx::properties
