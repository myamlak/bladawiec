// Population analyses: Mulliken [Mulliken1955], Lowdin
// [Lowdin1950], Mayer [Mayer1983], and Gopinathan-Jug [Gopinathan1983]
// bond orders - matrix algebra over the per-spin densities and the overlap
// matrix. The numerical traps are documented at each site below. The
// combined AnalyzePopulations path computes S and S^{1/2} ONCE and shares
// them across all four analyses (the S^{1/2}-sharing requirement).
#include "qcx/properties/populations.hpp"

#include "internal/tensor_to_eigen.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace qcx::properties {
namespace {

// The four analyses share one shape contract: square n x n matrices with a
// common n, and aoRanges partitioning [0, n).
qcx::Result<void> ValidateDensities(const Eigen::MatrixXd& overlap,
                                    const Eigen::MatrixXd& densityAlpha,
                                    const Eigen::MatrixXd& densityBeta,
                                    const std::vector<AoRange>& aoRanges) {
    const Eigen::Index n = overlap.rows();

    if (overlap.cols() != n || densityAlpha.rows() != n || densityAlpha.cols() != n ||
        densityBeta.rows() != n || densityBeta.cols() != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "overlap and densities must be square n x n with a common n"});
    }

    for (const AoRange& range : aoRanges)
    {
        if (range.firstFunction + range.functionCount > static_cast<std::size_t>(n))
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "aoRanges exceed the matrix dimension"});
        }
    }

    return {};
}

// Mulliken core over the precomputed P_sigma S products: the atomic
// population of atom a is the sum of the (P_sigma S) diagonal over the
// atom's functions; the orbital populations are the full diagonal.
MullikenPopulations ComputeMulliken(const Eigen::MatrixXd& psAlpha,
                                    const Eigen::MatrixXd& psBeta,
                                    const std::vector<AoRange>& aoRanges) {
    MullikenPopulations result;
    const Eigen::Index atomCount = static_cast<Eigen::Index>(aoRanges.size());
    result.alpha = Eigen::VectorXd::Zero(atomCount);
    result.beta = Eigen::VectorXd::Zero(atomCount);
    result.orbitalAlpha = psAlpha.diagonal();
    result.orbitalBeta = psBeta.diagonal();
    result.orbitalTotal = result.orbitalAlpha + result.orbitalBeta;

    for (std::size_t a = 0; a < aoRanges.size(); ++a)
    {
        const Eigen::Index first = static_cast<Eigen::Index>(aoRanges[a].firstFunction);
        const Eigen::Index count = static_cast<Eigen::Index>(aoRanges[a].functionCount);
        result.alpha(static_cast<Eigen::Index>(a)) = psAlpha.diagonal().segment(first, count).sum();
        result.beta(static_cast<Eigen::Index>(a)) = psBeta.diagonal().segment(first, count).sum();
    }

    result.total = result.alpha + result.beta;
    result.spin = result.alpha - result.beta;
    return result;
}

// Lowdin core over the precomputed forward root: P_ort = S^{1/2} P S^{1/2},
// atomic population = sum of the P_ort diagonal over the atom.
LowdinPopulations ComputeLowdin(const Eigen::MatrixXd& ortAlpha,
                                const Eigen::MatrixXd& ortBeta,
                                const std::vector<AoRange>& aoRanges) {
    LowdinPopulations result;
    const Eigen::Index atomCount = static_cast<Eigen::Index>(aoRanges.size());
    result.alpha = Eigen::VectorXd::Zero(atomCount);
    result.beta = Eigen::VectorXd::Zero(atomCount);

    for (std::size_t a = 0; a < aoRanges.size(); ++a)
    {
        const Eigen::Index first = static_cast<Eigen::Index>(aoRanges[a].firstFunction);
        const Eigen::Index count = static_cast<Eigen::Index>(aoRanges[a].functionCount);
        result.alpha(static_cast<Eigen::Index>(a)) =
            ortAlpha.diagonal().segment(first, count).sum();
        result.beta(static_cast<Eigen::Index>(a)) = ortBeta.diagonal().segment(first, count).sum();
    }

    result.total = result.alpha + result.beta;
    result.spin = result.alpha - result.beta;
    return result;
}

// Mayer core over the precomputed P_sigma S products. Bond orders: the
// b < a triangle only, mirrored. Free valence: the (PS_alpha - PS_beta)
// cross-product sum over the atom's own block - identically zero for RHF
// because the two spin densities coincide there (documented in the header).
MayerAnalysis ComputeMayer(const Eigen::MatrixXd& psAlpha,
                           const Eigen::MatrixXd& psBeta,
                           const std::vector<AoRange>& aoRanges) {
    MayerAnalysis result;
    const Eigen::Index atomCount = static_cast<Eigen::Index>(aoRanges.size());
    result.bondOrders = Eigen::MatrixXd::Zero(atomCount, atomCount);
    result.freeValences = Eigen::VectorXd::Zero(atomCount);

    for (std::size_t a = 0; a < aoRanges.size(); ++a)
    {
        const Eigen::Index aFirst = static_cast<Eigen::Index>(aoRanges[a].firstFunction);
        const Eigen::Index aCount = static_cast<Eigen::Index>(aoRanges[a].functionCount);

        for (std::size_t b = 0; b < a; ++b)
        {
            const Eigen::Index bFirst = static_cast<Eigen::Index>(aoRanges[b].firstFunction);
            const Eigen::Index bCount = static_cast<Eigen::Index>(aoRanges[b].functionCount);
            const Eigen::MatrixXd alphaBlock = psAlpha.block(aFirst, bFirst, aCount, bCount);
            // The Mayer pairing is PS(a_i, b_j) x PS(b_j, a_i): the (b, a)
            // block is the TRANSPOSE partner of the (a, b) block - the
            // full-matrix transpose mirror, NOT the transpose of the (a, b)
            // cross region (those coincide only when PS is symmetric).
            // PS = P S is not symmetric, so a region-transpose pairs
            // PS(a_i, b_j) with PS(b_i, a_j) and the bond order is wrong
            // wherever the cross block is rectangular: the O2/STO-3G 5x5
            // cross blocks gave 1.46 without any transpose and 1.6019685
            // with the region transpose (the interim numpy cross-check's
            // bug, 2026-08-25); the full-matrix pairing restores the true
            // Mayer order 2.0 pinned by populations_test. The RHF pins
            // stayed green throughout because their 1x5/1x1 cross blocks
            // are square, where the pairings coincide.
            const Eigen::MatrixXd alphaBlockT =
                psAlpha.block(bFirst, aFirst, bCount, aCount).transpose();
            const Eigen::MatrixXd betaBlock = psBeta.block(aFirst, bFirst, aCount, bCount);
            const Eigen::MatrixXd betaBlockT =
                psBeta.block(bFirst, aFirst, bCount, aCount).transpose();
            const double value = 2.0 * ((alphaBlock.array() * alphaBlockT.array()).sum() +
                                        (betaBlock.array() * betaBlockT.array()).sum());
            result.bondOrders(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) = value;
            result.bondOrders(static_cast<Eigen::Index>(b), static_cast<Eigen::Index>(a)) = value;
        }

        const Eigen::MatrixXd alphaBlock = psAlpha.block(aFirst, aFirst, aCount, aCount);
        const Eigen::MatrixXd betaBlock = psBeta.block(aFirst, aFirst, aCount, aCount);
        const Eigen::MatrixXd difference = alphaBlock - betaBlock;
        result.freeValences(static_cast<Eigen::Index>(a)) =
            (difference.array() * difference.transpose().array()).sum();
    }

    result.totalValences = result.freeValences;

    for (std::size_t a = 0; a < aoRanges.size(); ++a)
    {
        for (std::size_t b = 0; b < aoRanges.size(); ++b)
        {
            if (a != b)
            {
                result.totalValences(static_cast<Eigen::Index>(a)) +=
                    result.bondOrders(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b));
            }
        }
    }

    return result;
}

// Gopinathan-Jug core over the precomputed orthogonalized densities: the
// squared block sum over ALL pairs including the diagonal (the b loop runs
// over every atom - the self-term is a formula artifact, documented in the
// header).
GopinathanJugAnalysis ComputeGopinathanJug(const Eigen::MatrixXd& ortAlpha,
                                           const Eigen::MatrixXd& ortBeta,
                                           const std::vector<AoRange>& aoRanges) {
    GopinathanJugAnalysis result;
    const Eigen::Index atomCount = static_cast<Eigen::Index>(aoRanges.size());
    result.bondOrders = Eigen::MatrixXd::Zero(atomCount, atomCount);

    for (std::size_t a = 0; a < aoRanges.size(); ++a)
    {
        const Eigen::Index aFirst = static_cast<Eigen::Index>(aoRanges[a].firstFunction);
        const Eigen::Index aCount = static_cast<Eigen::Index>(aoRanges[a].functionCount);

        for (std::size_t b = 0; b < aoRanges.size(); ++b)
        {
            const Eigen::Index bFirst = static_cast<Eigen::Index>(aoRanges[b].firstFunction);
            const Eigen::Index bCount = static_cast<Eigen::Index>(aoRanges[b].functionCount);
            const Eigen::MatrixXd alphaBlock = ortAlpha.block(aFirst, bFirst, aCount, bCount);
            const Eigen::MatrixXd betaBlock = ortBeta.block(aFirst, bFirst, aCount, bCount);
            result.bondOrders(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) =
                alphaBlock.array().square().sum() + betaBlock.array().square().sum();
        }
    }

    return result;
}

} // namespace

qcx::Result<std::vector<AoRange>> AoIndexRangesByAtom(const qcx::molecule::Molecule& molecule,
                                                      const qcx::basisset::BasisSet& basisSet) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    // Shells are flattened in atom order (shell_pairs.cpp), so each atom's
    // functions form one contiguous range; accumulate per owning atom.
    std::vector<AoRange> ranges(molecule.AtomCount());

    for (const qcx::integrals::ShellInfo& shell : pairList->shells)
    {
        AoRange& range = ranges[shell.atomIndex];

        if (range.functionCount == 0)
        {
            range.firstFunction = shell.functionOffset;
        }

        range.functionCount += qcx::integrals::ShellFunctionCount(shell);
    }

    return ranges;
}

Eigen::MatrixXd SymmetricSquareRootOfOverlap(const Eigen::MatrixXd& overlap) {
    // The forward root: same proven SelfAdjointEigenSolver idiom as scf's
    // OrthogonalizeOverlap (scf_common.cpp), but .sqrt() instead of
    // .rsqrt() - S^{1/2}, not S^{-1/2} (the header's warning).
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(overlap);
    return solver.eigenvectors() * solver.eigenvalues().array().sqrt().matrix().asDiagonal() *
           solver.eigenvectors().transpose();
}

qcx::Result<MullikenPopulations> AnalyzeMulliken(const Eigen::MatrixXd& overlap,
                                                 const Eigen::MatrixXd& densityAlpha,
                                                 const Eigen::MatrixXd& densityBeta,
                                                 const std::vector<AoRange>& aoRanges) {
    auto validated = ValidateDensities(overlap, densityAlpha, densityBeta, aoRanges);

    if (!validated.has_value())
    {
        return std::unexpected(validated.error());
    }

    return ComputeMulliken(densityAlpha * overlap, densityBeta * overlap, aoRanges);
}

qcx::Result<LowdinPopulations> AnalyzeLowdin(const Eigen::MatrixXd& overlap,
                                             const Eigen::MatrixXd& densityAlpha,
                                             const Eigen::MatrixXd& densityBeta,
                                             const std::vector<AoRange>& aoRanges) {
    auto validated = ValidateDensities(overlap, densityAlpha, densityBeta, aoRanges);

    if (!validated.has_value())
    {
        return std::unexpected(validated.error());
    }

    const Eigen::MatrixXd root = SymmetricSquareRootOfOverlap(overlap);
    return ComputeLowdin(root * densityAlpha * root, root * densityBeta * root, aoRanges);
}

qcx::Result<MayerAnalysis> AnalyzeMayer(const Eigen::MatrixXd& overlap,
                                        const Eigen::MatrixXd& densityAlpha,
                                        const Eigen::MatrixXd& densityBeta,
                                        const std::vector<AoRange>& aoRanges) {
    auto validated = ValidateDensities(overlap, densityAlpha, densityBeta, aoRanges);

    if (!validated.has_value())
    {
        return std::unexpected(validated.error());
    }

    return ComputeMayer(densityAlpha * overlap, densityBeta * overlap, aoRanges);
}

qcx::Result<GopinathanJugAnalysis> AnalyzeGopinathanJug(const Eigen::MatrixXd& overlap,
                                                        const Eigen::MatrixXd& densityAlpha,
                                                        const Eigen::MatrixXd& densityBeta,
                                                        const std::vector<AoRange>& aoRanges) {
    auto validated = ValidateDensities(overlap, densityAlpha, densityBeta, aoRanges);

    if (!validated.has_value())
    {
        return std::unexpected(validated.error());
    }

    const Eigen::MatrixXd root = SymmetricSquareRootOfOverlap(overlap);
    return ComputeGopinathanJug(root * densityAlpha * root, root * densityBeta * root, aoRanges);
}

qcx::Result<PopulationAnalysis> AnalyzePopulations(const qcx::molecule::Molecule& molecule,
                                                   const qcx::basisset::BasisSet& basisSet,
                                                   const Eigen::MatrixXd& densityAlpha,
                                                   const Eigen::MatrixXd& densityBeta) {
    auto ranges = AoIndexRangesByAtom(molecule, basisSet);

    if (!ranges.has_value())
    {
        return std::unexpected(ranges.error());
    }

    auto overlapTensor = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlapTensor.has_value())
    {
        return std::unexpected(overlapTensor.error());
    }

    const Eigen::MatrixXd overlap = internal::TensorToEigen(*overlapTensor);
    auto validated = ValidateDensities(overlap, densityAlpha, densityBeta, *ranges);

    if (!validated.has_value())
    {
        return std::unexpected(validated.error());
    }

    // S and S^{1/2} are computed exactly once and shared by all four
    // analyses (the sharing requirement).
    const Eigen::MatrixXd root = SymmetricSquareRootOfOverlap(overlap);
    const Eigen::MatrixXd psAlpha = densityAlpha * overlap;
    const Eigen::MatrixXd psBeta = densityBeta * overlap;
    const Eigen::MatrixXd ortAlpha = root * densityAlpha * root;
    const Eigen::MatrixXd ortBeta = root * densityBeta * root;
    PopulationAnalysis result;
    result.mulliken = ComputeMulliken(psAlpha, psBeta, *ranges);
    result.lowdin = ComputeLowdin(ortAlpha, ortBeta, *ranges);
    result.mayer = ComputeMayer(psAlpha, psBeta, *ranges);
    result.gopinathanJug = ComputeGopinathanJug(ortAlpha, ortBeta, *ranges);
    return result;
}

} // namespace qcx::properties
