// The ETS-NOCV total-energy decomposition (nocv.hpp):
// [Mitoraj2009] pairs the extended transition state partitioning
//   E_mol - sum_i E_frag_i = E_elstat + E_Pauli + E_orb
// with the paired Natural Orbitals for Chemical Valence of the
// orthogonalized density difference Delta P = P_mol - P_orth. The fragment
// SCFs run at their molecular geometries in the FULL molecular basis set
// (the ETS convention), and the functional values come from the scf
// arbitrary-density evaluator (qcx/scf/density_energy.hpp), which
// reproduces the SCF loops' energies bit-for-bit at the converged
// densities - the decomposition's energy-identity pin.
#include "qcx/properties/nocv.hpp"

#include "internal/tensor_to_eigen.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/properties/populations.hpp"
#include "qcx/scf/density_energy.hpp"
#include "qcx/scf/uhf.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

namespace qcx::properties {

namespace {

// The Aufbau shell-degeneracy sequence of the isolated-atom state
// classification; the canonical copy lives in nalewajski.cpp (the two
// consumers share the same table - a dedup into an internal header is a
// follow-up candidate).
inline constexpr std::array<int, 19> kShellDegeneracies{
    1, 1, 3, 1, 3, 1, 5, 3, 1, 5, 3, 1, 7, 5, 3, 1, 7, 5, 3};

// The Hund ground-state multiplicity of one neutral atom (nalewajski.cpp's
// ClassifyFragmentStates rule: n_deg + 1 for a less-than-half-filled
// degenerate shell, else 2 n_states - n_deg + 1), Z <= 19.
qcx::Result<int> HundMultiplicity(int atomicNumber) {
    if (atomicNumber < 1 || atomicNumber > 19)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented, "the Aufbau table covers Z = 1..19"});
    }

    int electrons = atomicNumber;

    for (const int degeneracy : kShellDegeneracies)
    {
        if (electrons > 2 * degeneracy)
        {
            electrons -= 2 * degeneracy;
        } else
        {
            return electrons < degeneracy ? electrons + 1 : 2 * degeneracy - electrons + 1;
        }
    }

    // Unreachable for Z <= 19: the table holds 118 electrons.
    return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument, "unreachable"});
}

// One fragment's ETS embedding: its own SCF at its molecular geometry in
// the full molecular basis, plus the spin-summed frozen density, the
// per-spin occupied blocks that span the combined occupied space, and the
// fragment's own one-electron operators (its nuclear repulsion and core
// Hamiltonian feed the Coulomb-only fragment self-energy).
struct EtsFragment {
    double energy = 0.0; ///< The fragment SCF total energy E_frag_i.
    Eigen::MatrixXd density; ///< P_i = D_a + D_b (n x n, spin-summed).
    Eigen::MatrixXd occupiedAlpha; ///< n x nAlpha occupied alpha orbitals.
    Eigen::MatrixXd occupiedBeta; ///< n x nBeta occupied beta orbitals.
    qcx::molecule::Molecule molecule; ///< The fragment sub-molecule.
    Eigen::MatrixXd coreHamiltonian; ///< The fragment's own T + V (n x n).
};

qcx::Result<EtsFragment> RunEtsFragmentScf(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& overlap,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const std::vector<std::size_t>& atomIndices,
    const qcx::scf::UhfOptions& options) {
    const auto& atoms = molecule.Atoms();
    const auto& coordinates = molecule.CoordinatesBohr();

    std::vector<qcx::molecule::Atom> fragmentAtoms;
    auto fragmentCoordinates =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({atomIndices.size(), 3});

    if (!fragmentCoordinates.has_value())
    {
        return std::unexpected(fragmentCoordinates.error());
    }

    int electrons = 0;

    for (std::size_t row = 0; row < atomIndices.size(); ++row)
    {
        const auto& atom = atoms[atomIndices[row]];
        fragmentAtoms.push_back(atom);
        electrons += atom.atomicNumber;
        // The FRAGMENT's index is row; the coordinates are the molecule's,
        // keyed by the fragment's atom index (row == atomIndices[row] only
        // for the prefix fragments {0..k}).
        (*fragmentCoordinates)(row, 0) = coordinates(atomIndices[row], 0);
        (*fragmentCoordinates)(row, 1) = coordinates(atomIndices[row], 1);
        (*fragmentCoordinates)(row, 2) = coordinates(atomIndices[row], 2);
    }

    fragmentCoordinates->MarkHostDirty();

    // Single-atom fragments take their Hund ground-state multiplicity
    // (Z <= 19); multi-atom fragments the closed-shell convention (1 for
    // an even electron count, 2 for an odd one) - the ETS standard for
    // closed-shell decompositions.
    int multiplicity = 1;

    if (atomIndices.size() == 1)
    {
        auto hund = HundMultiplicity(atoms[atomIndices[0]].atomicNumber);

        if (!hund.has_value())
        {
            return std::unexpected(hund.error());
        }

        multiplicity = *hund;
    } else
    {
        multiplicity = electrons % 2 == 0 ? 1 : 2;
    }

    auto fragment = qcx::molecule::Molecule::Create(
        std::move(fragmentAtoms), std::move(*fragmentCoordinates), 0, multiplicity);

    if (!fragment.has_value())
    {
        return std::unexpected(fragment.error());
    }

    // The fragment's core in the FULL molecular basis: the kinetic matrix
    // spans all of the molecule's functions (the one-electron builders
    // derive the shell list from the molecule, so a fragment-only molecule
    // would yield a fragment-sized core), and the nuclear attraction sums
    // only the fragment's own centers.
    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet, atomIndices);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    Eigen::MatrixXd core = internal::TensorToEigen(*kinetic) + internal::TensorToEigen(*nuclear);

    // The fragment SCF: the MOLECULAR overlap and repulsion tensor (the
    // fragment orbitals span the whole space), the fragment's own core.
    auto scf = qcx::scf::RunUhfScf(*fragment, overlap, core, eri, options);

    if (!scf.has_value())
    {
        return std::unexpected(scf.error());
    }

    // Convergence gate: a non-converged run returns the
    // LAST iterate, not the fragment ground state. The ETS identity
    // E_mol - sum E_frag = E_elstat + E_Pauli + E_orb holds at any density
    // (the degradation is the physical accuracy of the fragment states
    // only), so without this gate a missed convergence would be silent.
    if (!scf->converged)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "a fragment SCF did not converge within its iteration budget; the ETS "
                       "decomposition needs converged fragment states"});
    }

    const int nBeta = (electrons - (multiplicity - 1)) / 2;
    const int nAlpha = (electrons + (multiplicity - 1)) / 2;

    // Molecule is move-only (no default ctor), so the fragment record is
    // aggregate-brace-constructed once its pieces are ready.
    return EtsFragment{scf->totalEnergy,
                       scf->densityAlpha + scf->densityBeta,
                       scf->coefficientsAlpha.leftCols(nAlpha),
                       scf->coefficientsBeta.leftCols(nBeta),
                       std::move(*fragment),
                       std::move(core)};
}

// One spin-channel resolution of a Lowdin density difference against the
// transition-state Fock matrix in the Lowdin basis: the sign-paired (+nu,
// -nu) eigenpairs contribute their own-sign diagonal channel energies
// lambda_+ F_++ + lambda_- F_-- with their own eigenvalues (the EXACT
// diagonal resolution of Tr[Delta D F^TS], E[P1] - E[P2] = Tr[(P1 - P2)
// F^TS] for the quadratic functional at the transition-state Fock, by the
// symmetry of G); an odd function count leaves the unpaired tail eigenvalue
// as its own channel, so the component sum is the full trace. The textbook
// form nu (F_++ - F_--), the DIAGONAL elements in the NOCV basis, follows
// only in the exactly-paired limit (equal-rank two-projector differences,
// the closed-shell fragments), where lambda_- = -lambda_+; the off-diagonal
// Mitoraj bilinear form nu F_-+ coincides only in that same limit and
// would leave a pairing-defect remainder in E_Pauli.
struct NocvChannelSet {
    Eigen::VectorXd components; ///< The per-channel energies (Hartree).
    Eigen::VectorXd eigenvalues; ///< The +nu_k of each pair (electrons).
    double energy = 0.0; ///< sum(components) = Tr[Delta D F^TS].
};

// (lowdinDelta, fockLowdin) are the Loewdin-transformed density difference
// and Fock matrix - distinct quantities, fixed order.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
NocvChannelSet ResolveChannels(const Eigen::MatrixXd& lowdinDelta,
                               const Eigen::MatrixXd& fockLowdin) {
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(lowdinDelta);
    const Eigen::Index n = lowdinDelta.rows();

    // The (+nu, -nu) pairs: the k-th largest POSITIVE eigenvalue pairs
    // with the k-th largest-magnitude NEGATIVE one - the exact
    // two-projector pairing, decided by the SIGN and not by a |lambda|
    // comparison: the paired magnitudes of an open-shell pairing defect
    // differ only at the 1e-15 level, and a magnitude-adjacent pairing
    // would be noise-dependent (the two orders of a +-nu tie land on
    // different channel values). The sign lists hold only the eigenvalues
    // clearly above the numerical zero floor: a ~0 residual (1e-15) is a
    // null-space artifact whose sign is solver noise, and a sign choice
    // for it would re-open the same noise-dependence (Eigen and numpy
    // hand the ~0s different signs). An imbalanced remainder (odd function
    // counts, unequal sign counts) pairs against the ~0 tail eigenvalues,
    // and the final leftover is its own tail channel, so the component
    // sum stays the full trace.
    const double zeroFloor = 1e-10 * solver.eigenvalues().cwiseAbs().maxCoeff();
    std::vector<Eigen::Index> positive, negative;

    for (Eigen::Index i = 0; i < n; ++i)
    {
        const double lambda = solver.eigenvalues()(i);

        if (std::abs(lambda) >= zeroFloor)
        {
            (lambda >= 0.0 ? positive : negative).push_back(i);
        }
    }

    std::sort(positive.begin(), positive.end(), [&solver](Eigen::Index a, Eigen::Index b) {
        return solver.eigenvalues()(a) > solver.eigenvalues()(b);
    });
    std::sort(negative.begin(), negative.end(), [&solver](Eigen::Index a, Eigen::Index b) {
        return std::abs(solver.eigenvalues()(a)) > std::abs(solver.eigenvalues()(b));
    });

    // The partners of the post-core pairs: the eigenvalues outside the
    // sign core - the extras of the longer sign list and the ~0 tail -
    // by magnitude ascending. (The k-th largest positive of a n_alpha vs
    // n_beta count imbalance is an EXTRA, not a core element: it pairs
    // against the smallest leftover, exactly as the closed-shell spectra's
    // ~0 tail does.)
    const std::size_t core = std::min(positive.size(), negative.size());
    std::vector<Eigen::Index> partners;
    std::vector<bool> inCore(static_cast<std::size_t>(n), false);

    for (std::size_t k = 0; k < core; ++k)
    {
        inCore[static_cast<std::size_t>(positive[k])] = true;
        inCore[static_cast<std::size_t>(negative[k])] = true;
    }

    for (Eigen::Index i = 0; i < n; ++i)
    {
        if (!inCore[static_cast<std::size_t>(i)])
        {
            partners.push_back(i);
        }
    }

    std::sort(partners.begin(), partners.end(), [&solver](Eigen::Index a, Eigen::Index b) {
        return std::abs(solver.eigenvalues()(a)) < std::abs(solver.eigenvalues()(b));
    });

    const std::size_t pairCount = static_cast<std::size_t>(n) / 2;
    const std::size_t channelCount = pairCount + static_cast<std::size_t>(n) % 2;
    NocvChannelSet set;
    set.components.resize(static_cast<Eigen::Index>(channelCount));
    set.eigenvalues.resize(static_cast<Eigen::Index>(channelCount));

    std::size_t partnerCursor = 0;
    const auto takePartner = [&]() { return partners[partnerCursor++]; };

    for (std::size_t pair = 0; pair < pairCount; ++pair)
    {
        // A post-core pair takes the next two leftover eigenvalues, the
        // larger one carrying the + role (a pair may be (extra, ~0),
        // (extra, extra), or (~0, ~0); a ~0 may carry either sign - the
        // comparison decides the role, never the sign).
        Eigen::Index iPlus = pair < core ? positive[pair] : takePartner();
        Eigen::Index iMinus = pair < core ? negative[pair] : takePartner();

        if (pair >= core && solver.eigenvalues()(iPlus) < solver.eigenvalues()(iMinus))
        {
            std::swap(iPlus, iMinus);
        }

        const double lambdaPlus = solver.eigenvalues()(iPlus);
        const double lambdaMinus = solver.eigenvalues()(iMinus);

        // The channel energy: the pair's contribution to Tr[Delta D F^TS]
        // with each eigenpair's own diagonal element and eigenvalue:
        // lambda_+ F_++ + lambda_- F_-- (see the function comment).
        const double channel = lambdaPlus * solver.eigenvectors().col(iPlus).dot(
                                                fockLowdin * solver.eigenvectors().col(iPlus)) +
                               lambdaMinus * solver.eigenvectors().col(iMinus).dot(
                                                 fockLowdin * solver.eigenvectors().col(iMinus));
        set.components(static_cast<Eigen::Index>(pair)) = channel;
        set.eigenvalues(static_cast<Eigen::Index>(pair)) = lambdaPlus;
        set.energy += channel;
    }

    // An odd function count leaves one unpaired eigenvalue (exactly zero
    // for the paired closed-shell spectra); it forms its own channel so
    // the component sum stays the full trace.
    if (n % 2 == 1)
    {
        const Eigen::Index tailIndex = takePartner();
        const double lambdaTail = solver.eigenvalues()(tailIndex);
        const double tailChannel =
            lambdaTail * solver.eigenvectors().col(tailIndex).dot(
                             fockLowdin * solver.eigenvectors().col(tailIndex));
        set.components(static_cast<Eigen::Index>(pairCount)) = tailChannel;
        set.eigenvalues(static_cast<Eigen::Index>(pairCount)) = lambdaTail;
        set.energy += tailChannel;
    }

    return set;
}

// Concatenates the per-fragment occupied blocks horizontally: the
// combined occupied space of the frozen fragment determinant.
Eigen::MatrixXd JoinBlocks(const std::vector<Eigen::MatrixXd>& blocks, Eigen::Index rows) {
    Eigen::Index totalColumns = 0;

    for (const auto& block : blocks)
    {
        totalColumns += block.cols();
    }

    Eigen::MatrixXd joined = Eigen::MatrixXd::Zero(rows, totalColumns);
    Eigen::Index column = 0;

    for (const auto& block : blocks)
    {
        joined.block(0, column, rows, block.cols()) = block;
        column += block.cols();
    }

    return joined;
}

} // namespace

qcx::Result<NocvEtsDecomposition> AnalyzeNocvEts(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& density,
    const std::vector<std::vector<std::size_t>>& fragments,
    const qcx::scf::UhfOptions& fragmentOptions) {
    const std::size_t n = eri.Shape()[0];

    if (eri.Shape()[1] != n || eri.Shape()[2] != n || eri.Shape()[3] != n ||
        static_cast<std::size_t>(coreHamiltonian.rows()) != n ||
        static_cast<std::size_t>(coreHamiltonian.cols()) != n ||
        static_cast<std::size_t>(density.rows()) != n ||
        static_cast<std::size_t>(density.cols()) != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "core Hamiltonian and density must match the ERI function count"});
    }

    const std::size_t atomCount = molecule.AtomCount();

    // The fragment partition must be disjoint and cover every atom once.
    std::vector<bool> covered(atomCount, false);

    for (const auto& group : fragments)
    {
        for (const std::size_t atom : group)
        {
            if (atom >= atomCount || covered[atom])
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument,
                               "fragments must be disjoint groups of in-range atom indices"});
            }

            covered[atom] = true;
        }
    }

    for (const bool atom : covered)
    {
        if (!atom)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "fragments must cover every atom exactly once"});
        }
    }

    // The molecular overlap, shared by every fragment SCF and the NOCV
    // orthogonalization.
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    const Eigen::MatrixXd overlapEigen = internal::TensorToEigen(*overlap);

    // The per-fragment SCFs, their frozen densities, and the E_J-only
    // fragment self-energies.
    NocvEtsDecomposition result;
    result.fragmentEnergies.resize(static_cast<Eigen::Index>(fragments.size()));
    std::vector<EtsFragment> fragmentScfs;
    fragmentScfs.reserve(fragments.size());
    std::vector<Eigen::MatrixXd> alphaBlocks;
    std::vector<Eigen::MatrixXd> betaBlocks;
    alphaBlocks.reserve(fragments.size());
    betaBlocks.reserve(fragments.size());
    Eigen::MatrixXd fragmentDensity =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    double fragmentEnergySum = 0.0;
    double fragmentCoulombEnergySum = 0.0;

    for (std::size_t i = 0; i < fragments.size(); ++i)
    {
        auto scf =
            RunEtsFragmentScf(molecule, basisSet, overlapEigen, eri, fragments[i], fragmentOptions);

        if (!scf.has_value())
        {
            return std::unexpected(scf.error());
        }

        result.fragmentEnergies(static_cast<Eigen::Index>(i)) = scf->energy;
        fragmentEnergySum += scf->energy;
        fragmentDensity += scf->density;
        alphaBlocks.push_back(scf->occupiedAlpha);
        betaBlocks.push_back(scf->occupiedBeta);

        auto coulomb = qcx::scf::EvaluateRhfDensityEnergy(
            scf->molecule, scf->coreHamiltonian, eri, scf->density, /*includeExchange=*/false);

        if (!coulomb.has_value())
        {
            return std::unexpected(coulomb.error());
        }

        fragmentCoulombEnergySum += coulomb->totalEnergy;
        fragmentScfs.push_back(std::move(*scf));
    }

    result.fragmentDensity = std::move(fragmentDensity);

    // E_elstat: the classical Coulomb-only interaction of the frozen
    // fragment densities - the exchange-free functional value of the
    // combined density minus the fragments' own. The one-electron cross
    // terms, the inter-fragment J, and the inter-fragment nuclear
    // repulsion all survive the difference exactly, so this IS the
    // classical electrostatic interaction.
    auto combinedCoulomb = qcx::scf::EvaluateRhfDensityEnergy(
        molecule, coreHamiltonian, eri, result.fragmentDensity, /*includeExchange=*/false);

    if (!combinedCoulomb.has_value())
    {
        return std::unexpected(combinedCoulomb.error());
    }

    result.electrostatic = combinedCoulomb->totalEnergy - fragmentCoulombEnergySum;

    // P_orth: the per-spin-channel projector onto the combined occupied
    // space, C_occ (C_occ^T S C_occ)^-1 C_occ^T with NO per-channel factor.
    // The spin-summed density convention of 2 P appears only as the SUM of
    // two equal closed-shell channels (2 P = P_a + P_b when P_a = P_b); a
    // per-channel 2 would double the open-shell density difference and
    // break the H2/STO-3G limit (two doublets spanning the full space give
    // P_orth = 2 S^-1 = D_mol and E_orb = 0 exactly).
    const Eigen::MatrixXd combinedAlpha = JoinBlocks(alphaBlocks, static_cast<Eigen::Index>(n));
    const Eigen::MatrixXd combinedBeta = JoinBlocks(betaBlocks, static_cast<Eigen::Index>(n));
    const Eigen::MatrixXd overlapAlpha = combinedAlpha.transpose() * overlapEigen * combinedAlpha;
    const Eigen::MatrixXd overlapBeta = combinedBeta.transpose() * overlapEigen * combinedBeta;
    const Eigen::MatrixXd projectorAlpha =
        combinedAlpha * overlapAlpha.inverse() * combinedAlpha.transpose();
    const Eigen::MatrixXd projectorBeta =
        combinedBeta * overlapBeta.inverse() * combinedBeta.transpose();
    result.orthogonalizedDensityAlpha = projectorAlpha;
    result.orthogonalizedDensityBeta = projectorBeta;
    result.orthogonalizedDensity = projectorAlpha + projectorBeta;

    // E_Pauli: what the antisymmetrized, orthogonalized fragment
    // determinant adds beyond the frozen combination's classical
    // interaction (defined so the partition identity holds exactly).
    auto orthEnergy = qcx::scf::EvaluateRhfDensityEnergy(
        molecule, coreHamiltonian, eri, result.orthogonalizedDensity);

    if (!orthEnergy.has_value())
    {
        return std::unexpected(orthEnergy.error());
    }

    result.pauli = orthEnergy->totalEnergy - fragmentEnergySum - result.electrostatic;

    // The NOCV channels of Delta P = P_mol - P_orth: the EXACT resolution
    // E[P_mol] - E[P_orth] = Tr[Delta P F^TS] of the quadratic RHF
    // functional (E[P1] - E[P2] = Tr[(P1 - P2) F^TS] with the
    // transition-state Fock matrix F^TS evaluated at 1/2(P1 + P2), by the
    // symmetry of G = J - 1/2 K). Every eigenpair of the Lowdin
    // orthogonalized difference contributes its own diagonal element.
    const Eigen::MatrixXd deltaP = density - result.orthogonalizedDensity;
    const Eigen::MatrixXd overlapRoot = SymmetricSquareRootOfOverlap(overlapEigen);
    const Eigen::MatrixXd overlapRootInverse = overlapRoot.inverse();
    const Eigen::MatrixXd deltaPLowdin = overlapRoot * deltaP * overlapRoot;

    auto transitionFock = qcx::scf::EvaluateRhfDensityEnergy(
        molecule, coreHamiltonian, eri, 0.5 * (result.orthogonalizedDensity + density));

    if (!transitionFock.has_value())
    {
        return std::unexpected(transitionFock.error());
    }

    // The transition-state Fock matrix in the Lowdin basis, where the
    // channel diagonal elements read v^T F^TS v directly.
    const Eigen::MatrixXd fockLowdin =
        overlapRootInverse * transitionFock->fock * overlapRootInverse;
    const auto channels = ResolveChannels(deltaPLowdin, fockLowdin);
    result.orbitalComponents = channels.components;
    result.nocvEigenvalues = channels.eigenvalues;
    result.orbital = channels.energy;

    // The UNRESTRICTED (spin-resolved) resolution of the same relaxation.
    // The closed-shell molecular density splits D_sigma = D/2 per spin, so
    // the per-spin density differences read Delta D_sigma = D/2 -
    // P_orth,sigma, and the per-spin transition Focks
    // F_sigma^TS = h + J[P^TS_total] - K[D_sigma^TS] at the per-spin
    // transition densities D_sigma^TS = 1/2(P_orth,sigma + D/2) are the
    // UHF FockBuilder values at (D_a^TS, D_b^TS): the J argument sums to
    // the TOTAL transition density 1/2(P_orth + D) by construction, and
    // each spin's exchange contracts its own block. The spin-resolved
    // identity E[P_mol] - E[P_orth] = sum_sigma Tr[Delta D_sigma
    // F_sigma^TS] is exact (the symmetry of G = J - K per spin on the
    // total-density J), and the open-shell correction over the restricted
    // value reads 1/4 Tr[(P_a - P_b) K[P_a - P_b]] (the two functionals
    // coincide on spin-symmetric densities).
    const Eigen::MatrixXd molecularAlpha = 0.5 * density;
    const Eigen::MatrixXd molecularBeta = 0.5 * density;
    const Eigen::MatrixXd deltaAlpha = molecularAlpha - result.orthogonalizedDensityAlpha;
    const Eigen::MatrixXd deltaBeta = molecularBeta - result.orthogonalizedDensityBeta;
    const Eigen::MatrixXd transitionAlpha =
        0.5 * (result.orthogonalizedDensityAlpha + molecularAlpha);
    const Eigen::MatrixXd transitionBeta = 0.5 * (result.orthogonalizedDensityBeta + molecularBeta);

    auto transitionUf = qcx::scf::EvaluateUhfDensityEnergy(
        molecule, coreHamiltonian, eri, transitionAlpha, transitionBeta);

    if (!transitionUf.has_value())
    {
        return std::unexpected(transitionUf.error());
    }

    const Eigen::MatrixXd fockAlphaLowdin =
        overlapRootInverse * transitionUf->fockAlpha * overlapRootInverse;
    const Eigen::MatrixXd fockBetaLowdin =
        overlapRootInverse * transitionUf->fockBeta * overlapRootInverse;
    const auto alphaChannels =
        ResolveChannels(overlapRoot * deltaAlpha * overlapRoot, fockAlphaLowdin);
    const auto betaChannels =
        ResolveChannels(overlapRoot * deltaBeta * overlapRoot, fockBetaLowdin);
    result.orbitalComponentsAlpha = alphaChannels.components;
    result.orbitalComponentsBeta = betaChannels.components;
    result.nocvEigenvaluesAlpha = alphaChannels.eigenvalues;
    result.nocvEigenvaluesBeta = betaChannels.eigenvalues;
    result.orbitalUnrestricted = alphaChannels.energy + betaChannels.energy;

    result.bindingEnergy = result.electrostatic + result.pauli + result.orbital;
    return result;
}

} // namespace qcx::properties
