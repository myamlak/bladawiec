// Nalewajski-Mrozek quadratic valence indices [Nalewajski1996]: Nalewajski,
// Mrozek & Mazur, "Quantum chemical valence indices from the
// one-determinantal difference approach", Can. J. Chem. 74, 1121-1130
// (1996), doi 10.1139/v96-126.
//
// The indices compare the simultaneous probability of finding two electrons
// on given atoms in the molecule and in the separated-atoms-limit (SAL)
// reference. In the one-determinantal (UHF) approximation that difference
// reduces to a quadratic form in the displacement of the charge-and-bond-
// order (CBO) matrix, Delta P = P(molecule) - P0(SAL): the diagonal
// displacements carry the ionic indices, the off-diagonal displacements the
// covalent indices, each accumulated as a squared displacement and summed
// over both spin channels.
//
// The reference state is the frozen SAL: every atom is described by its own
// isolated-atom UHF ground determinant, computed here on demand per distinct
// element in the molecular basis set. The open valence shell of that
// determinant is degenerate, so the SAL is additionally averaged over the
// shell's configurations; the accumulation below derives that average.
//
// Working basis. The CBO matrices are read in an orthogonalized AO basis:
// the molecular one in the Lowdin basis of the molecular overlap (the
// Gopinathan-Jug forward root shared with populations.hpp and eddb.hpp),
// each fragment one in the Lowdin basis of the atom's own overlap block.
// The frozen SAL density is a direct sum of isolated-fragment densities, so
// it has no element linking two different fragments; the two-centre indices
// therefore involve the molecular CBO matrix alone.
//
// Sign convention. The paper's bonding valences are negative. The
// accumulation here keeps the squared displacements and so reports
// magnitudes throughout; the one-centre division that turns the diatomic
// components into effective bond orders (the paper's Scheme III) adds each
// atom's one-centre share to the diatomic term.
#include "qcx/properties/nalewajski.hpp"

#include "internal/tensor_to_eigen.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/properties/populations.hpp"
#include "qcx/scf/uhf.hpp"

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <map>
#include <utility>
#include <vector>

namespace qcx::properties {
namespace {

// The occupied shells of a neutral atom in its ground configuration, in the
// order the Madelung (n + l, then n) rule fills them, each with its (2l + 1)
// spatial degeneracy: 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s 5f
// 6d 7p.
//
// The rule orders by n + l and breaks ties by n, which is what puts 4s
// before 3d: the principal-quantum-number order classifies Z <= 18
// identically but disagrees from Z = 19 (Madelung leaves 4s partly filled
// where the n-order would leave 3d). The nineteen entries carry 118
// electrons, so every Z <= 19 atom's configuration follows from this table;
// heavier elements need shells the table does not reach (kUnimplemented).
inline constexpr std::array<int, 19> kShellDegeneracies{
    1, 1, 3, 1, 3, 1, 5, 3, 1, 5, 3, 1, 7, 5, 3, 1, 7, 5, 3};

// The occupied-shell structure of a neutral atom: how many closed-shell
// core states (doubly occupied spatial orbitals) sit below the valence
// shell, and what the partly filled valence shell holds - how many
// degenerate states it offers and how many electrons occupy them.
struct FragmentStates {
    int coreStates = 0;
    int degenerateStates = 0;
    int degenerateElectrons = 0;
};

// Fills kShellDegeneracies in turn: a shell absorbs up to 2 x its degeneracy
// electrons, and the first shell that cannot be filled completely is the
// partly filled valence shell. Every shell below it is closed.
qcx::Result<FragmentStates> ClassifyFragmentStates(int atomicNumber) {
    if (atomicNumber < 1 || atomicNumber > 19)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                          "the Nalewajski-Mrozek Aufbau table covers Z = 1..19"});
    }

    int electrons = atomicNumber;

    for (const int degeneracy : kShellDegeneracies)
    {
        if (electrons > 2 * degeneracy)
        {
            electrons -= 2 * degeneracy;
        } else
        {
            return FragmentStates{(atomicNumber - electrons) / 2, degeneracy, electrons};
        }
    }

    // Unreachable for Z <= 19: the table holds 118 electrons.
    return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument, "unreachable"});
}

// The isolated-atom fragment SCF of one element: one neutral atom at the
// origin in its ground-state multiplicity, run in the MOLECULAR basis set -
// the same single-element-atom construction the SAD guess uses. The fragment
// orbitals are the converged alpha channel, columns ascending by orbital
// energy; the fragment overlap feeds the fragment Lowdin transform.
struct FragmentScf {
    Eigen::MatrixXd coefficients; ///< nZ x nZ fragment alpha orbitals.
    FragmentStates states;
};

qcx::Result<FragmentScf> RunFragmentScf(const qcx::molecule::Molecule& molecule,
                                        const qcx::basisset::BasisSet& basisSet,
                                        const qcx::molecule::Atom& atom) {
    auto states = ClassifyFragmentStates(atom.atomicNumber);

    if (!states.has_value())
    {
        return std::unexpected(states.error());
    }

    // The ground term of the isolated atom follows Hund's maximum-
    // multiplicity rule: a shell holding at most half its capacity leaves
    // every one of its electrons unpaired (S = n_deg / 2, multiplicity
    // n_deg + 1), a shell holding more than half fills the alpha channel
    // and puts the surplus in the beta channel, leaving
    // 2 n_deg_states - n_deg_electrons unpaired (multiplicity
    // 2 n_deg_states - n_deg_electrons + 1).
    const int multiplicity = states->degenerateElectrons < states->degenerateStates
                                 ? states->degenerateElectrons + 1
                                 : 2 * states->degenerateStates - states->degenerateElectrons + 1;

    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({1, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    coordinates->MarkHostDirty();

    auto fragment = qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{atom}, std::move(*coordinates), 0, multiplicity);

    if (!fragment.has_value())
    {
        return std::unexpected(fragment.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*fragment, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(*fragment, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*fragment, basisSet);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    auto eri = qcx::integrals::BuildEriTensorGeneral(*fragment, basisSet);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    auto scf =
        qcx::scf::RunUhfScf(*fragment,
                            internal::TensorToEigen(*overlap),
                            internal::TensorToEigen(*kinetic) + internal::TensorToEigen(*nuclear),
                            *eri);

    if (!scf.has_value())
    {
        return std::unexpected(scf.error());
    }

    return FragmentScf{std::move(scf->coefficientsAlpha), *states};
}

// All C(n, k) index combinations: every way to choose k of the n degenerate
// states. The order is irrelevant - the configurations enter the average
// with equal weights.
void BuildCombinations(
    int n, int k, std::vector<int>& current, int start, std::vector<std::vector<int>>& out) {
    if (static_cast<int>(current.size()) == k)
    {
        out.push_back(current);
        return;
    }

    for (int i = start; i + (k - static_cast<int>(current.size())) <= n; ++i)
    {
        current.push_back(i);
        BuildCombinations(n, k, current, i + 1, out);
        current.pop_back();
    }
}

std::vector<std::vector<int>> BuildCombinations(int n, int k) {
    std::vector<std::vector<int>> combinations;
    std::vector<int> current;

    if (k <= n)
    {
        BuildCombinations(n, k, current, 0, combinations);
    }

    return combinations;
}

// The one-center valence accumulation of one atom: the config-averaged
// ionic (diagonal) and covalent (off-diagonal) shares v_i_a and v_c_a of the
// CBO displacement Delta = P1f - P0f, where P0f is the fragment density in
// the fragment's own Lowdin basis (the frozen SAL, each fragment
// orthogonalized separately).
struct OneCenterValences {
    double covalent = 0.0; ///< v_c_a.
    double ionic = 0.0; ///< v_i_a.
};

// The one-center quadratic valence indices of one atom, averaged over the
// configurations of its degenerate valence shell.
//
// Composition of the average. The isolated-atom determinant of the previous
// step fixes the closed core and the shell's electron count, but a
// less-than-half-filled shell leaves the SCF free to span the degenerate
// states any way it likes, and that choice is not physical: it would leak
// into the SAL density and hence into the indices. The requirement the SAL
// must meet is therefore invariance under rotations of the degenerate shell
// among its own states - the reference density has to be the same whichever
// spanning set the fragment SCF happened to converge to. The degenerate
// shell has no preferred basis of its own, so the only density the shell can
// build from its own states that respects that requirement is the equal-
// weight average over its determinations: every way the shell's electrons
// can be placed in its states, at the term's fixed multiplicity, counted
// once. This averaging rule is derived from the invariance requirement, not
// taken from a published formula.
//
// The placement itself is the term's, not a free choice: Hund's maximum-
// multiplicity rule puts the shell's first n_deg_electrons into distinct
// states of one spin channel (alpha), so a shell no more than half full
// varies its alpha occupations and holds no beta electrons, while a shell
// more than half full fills alpha completely and varies its beta
// occupations. The channel that does not vary contributes its single
// placement to every configuration, so the configuration count is the
// varying channel's count.
//
// (p1fAlpha, p1fBeta, fragmentOverlap) are the alpha/beta P1F blocks and
// the fragment overlap - distinct quantities, fixed order.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
qcx::Result<OneCenterValences> OneCenterDifferences(const Eigen::MatrixXd& p1fAlpha,
                                                    const Eigen::MatrixXd& p1fBeta,
                                                    const Eigen::MatrixXd& fragmentOverlap,
                                                    const FragmentScf& fragment) {
    const int n = static_cast<int>(fragment.coefficients.rows());
    const int nCore = fragment.states.coreStates;
    const int nDegenerate = fragment.states.degenerateStates;
    const int nDegenerateElectrons = fragment.states.degenerateElectrons;

    const int alphaOccupied =
        nDegenerateElectrons <= nDegenerate ? nDegenerateElectrons : nDegenerate;
    const int betaOccupied = nDegenerateElectrons - alphaOccupied;
    const bool alphaVaries = nDegenerateElectrons <= nDegenerate;
    const std::vector<std::vector<int>> alphaConfigs =
        BuildCombinations(nDegenerate, alphaOccupied);
    const std::vector<std::vector<int>> betaConfigs = BuildCombinations(nDegenerate, betaOccupied);
    const std::vector<std::vector<int>>& varyingConfigs = alphaVaries ? alphaConfigs : betaConfigs;
    const std::vector<int>& fixedStates = alphaVaries ? betaConfigs.front() : alphaConfigs.front();
    const std::size_t configCount = varyingConfigs.size();

    // The closed core is common to every configuration and to both spin
    // channels (in the AO basis; P_core = C_core C_core^T).
    Eigen::MatrixXd coreDensity = Eigen::MatrixXd::Zero(n, n);

    for (int i = 0; i < nCore; ++i)
    {
        coreDensity.noalias() +=
            fragment.coefficients.col(i) * fragment.coefficients.col(i).transpose();
    }

    const Eigen::MatrixXd root = SymmetricSquareRootOfOverlap(fragmentOverlap);
    double covalent = 0.0;
    double ionic = 0.0;

    for (std::size_t config = 0; config < configCount; ++config)
    {
        // The configuration's fragment density per spin, transformed into
        // the fragment's Lowdin basis (P0f = S_f^{1/2} P S_f^{1/2}).
        Eigen::MatrixXd alphaDensity = coreDensity;

        for (const int i : alphaVaries ? varyingConfigs[config] : fixedStates)
        {
            alphaDensity.noalias() += fragment.coefficients.col(nCore + i) *
                                      fragment.coefficients.col(nCore + i).transpose();
        }

        Eigen::MatrixXd betaDensity = coreDensity;

        for (const int i : alphaVaries ? fixedStates : varyingConfigs[config])
        {
            betaDensity.noalias() += fragment.coefficients.col(nCore + i) *
                                     fragment.coefficients.col(nCore + i).transpose();
        }

        const Eigen::MatrixXd dAlpha = p1fAlpha - root * alphaDensity * root;
        const Eigen::MatrixXd dBeta = p1fBeta - root * betaDensity * root;

        // The one-center quadratic valence of a configuration is half the
        // sum of the squares of the CBO displacement's matrix elements,
        // summed over both spin channels. Written as an ordered-pair sum,
        //   (1/2) sum_sigma sum_{i, j in A} (Delta P^sigma_ij)^2,
        // the diagonal states occur once and every off-diagonal pair {i, j}
        // occurs twice, as (i, j) and as (j, i); halving therefore gives the
        // diagonal elements their 1/2 weight and leaves each off-diagonal
        // pair with a single weight. The diagonal elements carry the ionic
        // indices and the off-diagonal elements the covalent ones, so the
        // same halving splits the block's quadratic valence into its ionic
        // and covalent shares.
        const double diagonalShare =
            0.5 * (dAlpha.diagonal().squaredNorm() + dBeta.diagonal().squaredNorm());
        const double blockValence = 0.5 * (dAlpha.squaredNorm() + dBeta.squaredNorm());
        ionic += diagonalShare;
        covalent += blockValence - diagonalShare;
    }

    return OneCenterValences{covalent / static_cast<double>(configCount),
                             ionic / static_cast<double>(configCount)};
}

} // namespace

qcx::Result<NalewajskiBondOrders> AnalyzeNalewajskiBondOrders(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta) {
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

    // The molecular CBO matrix in the Lowdin basis (the Gopinathan-Jug
    // forward root shared with populations.hpp / eddb.cpp): P1 = S^{1/2} P
    // S^{1/2}.
    const Eigen::MatrixXd root = SymmetricSquareRootOfOverlap(internal::TensorToEigen(*overlap));
    const Eigen::MatrixXd p1Alpha = root * densityAlpha * root;
    const Eigen::MatrixXd p1Beta = root * densityBeta * root;

    const std::size_t nAtoms = aoRanges->size();
    const auto& atoms = molecule.Atoms();
    std::map<int, FragmentScf> fragments;
    std::vector<OneCenterValences> oneCenter(nAtoms);

    for (std::size_t a = 0; a < nAtoms; ++a)
    {
        auto fragment = fragments.find(atoms[a].atomicNumber);

        if (fragment == fragments.end())
        {
            auto built = RunFragmentScf(molecule, basisSet, atoms[a]);

            if (!built.has_value())
            {
                return std::unexpected(built.error());
            }

            fragment = fragments.emplace(atoms[a].atomicNumber, std::move(*built)).first;
        }

        const std::size_t first = (*aoRanges)[a].firstFunction;
        const std::size_t count = (*aoRanges)[a].functionCount;
        const Eigen::MatrixXd p1fAlpha = p1Alpha.block(static_cast<Eigen::Index>(first),
                                                       static_cast<Eigen::Index>(first),
                                                       static_cast<Eigen::Index>(count),
                                                       static_cast<Eigen::Index>(count));
        const Eigen::MatrixXd p1fBeta = p1Beta.block(static_cast<Eigen::Index>(first),
                                                     static_cast<Eigen::Index>(first),
                                                     static_cast<Eigen::Index>(count),
                                                     static_cast<Eigen::Index>(count));

        // The fragment's own Lowdin root uses the atom's overlap block: the
        // frozen SAL orthogonalizes each fragment separately.
        const Eigen::MatrixXd fragmentOverlap =
            internal::TensorToEigen(*overlap).block(static_cast<Eigen::Index>(first),
                                                    static_cast<Eigen::Index>(first),
                                                    static_cast<Eigen::Index>(count),
                                                    static_cast<Eigen::Index>(count));

        auto valences = OneCenterDifferences(p1fAlpha, p1fBeta, fragmentOverlap, fragment->second);

        if (!valences.has_value())
        {
            return std::unexpected(valences.error());
        }

        oneCenter[a] = *valences;
    }

    // The two-centre (diatomic) quadratic contributions v_ab: the molecular
    // CBO matrix's off-diagonal blocks, squared and summed over both spin
    // channels. No SAL term appears - the two functions live on different
    // fragments, and the frozen SAL has no block between two fragments.
    Eigen::MatrixXd diatomic =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nAtoms), static_cast<Eigen::Index>(nAtoms));

    for (std::size_t a = 0; a < nAtoms; ++a)
    {
        for (std::size_t b = a + 1; b < nAtoms; ++b)
        {
            double value = 0.0;
            const std::size_t firstA = (*aoRanges)[a].firstFunction;
            const std::size_t firstB = (*aoRanges)[b].firstFunction;

            for (std::size_t mu = 0; mu < (*aoRanges)[a].functionCount; ++mu)
            {
                for (std::size_t nu = 0; nu < (*aoRanges)[b].functionCount; ++nu)
                {
                    const Eigen::Index i = static_cast<Eigen::Index>(firstA + mu);
                    const Eigen::Index j = static_cast<Eigen::Index>(firstB + nu);
                    value += p1Alpha(i, j) * p1Alpha(i, j) + p1Beta(i, j) * p1Beta(i, j);
                }
            }

            diatomic(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) = value;
            diatomic(static_cast<Eigen::Index>(b), static_cast<Eigen::Index>(a)) = value;
        }
    }

    // The interatomic valence of an atom: the denominator the one-centre
    // division shares out.
    const Eigen::VectorXd total = diatomic.rowwise().sum();

    // The one-centre division of the one-center valences (the paper's
    // Scheme III): a one-centre displacement belongs to no single bond, so
    // to recover effective bond orders it is divided over the atom's bonds
    // in proportion to each bond's share v_ab / total_ab of that atom's
    // interatomic valence. Both ends of a bond contribute their
    // own divided share, so the diatomic component is scaled by
    // 1 + (v_i_a + v_c_a)/total_ab(a) + (v_i_b + v_c_b)/total_ab(b).
    // An atom with no interatomic valence has no bonds to divide over and
    // contributes no weight (its v_ab vanish, so its bond orders are zero
    // regardless).
    Eigen::MatrixXd bondOrders =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nAtoms), static_cast<Eigen::Index>(nAtoms));

    for (std::size_t a = 0; a < nAtoms; ++a)
    {
        for (std::size_t b = a + 1; b < nAtoms; ++b)
        {
            const double weightA = total(static_cast<Eigen::Index>(a)) > 0.0
                                       ? (oneCenter[a].ionic + oneCenter[a].covalent) /
                                             total(static_cast<Eigen::Index>(a))
                                       : 0.0;
            const double weightB = total(static_cast<Eigen::Index>(b)) > 0.0
                                       ? (oneCenter[b].ionic + oneCenter[b].covalent) /
                                             total(static_cast<Eigen::Index>(b))
                                       : 0.0;
            const double value =
                diatomic(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) *
                (1.0 + weightA + weightB);
            bondOrders(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) = value;
            bondOrders(static_cast<Eigen::Index>(b), static_cast<Eigen::Index>(a)) = value;
        }
    }

    NalewajskiBondOrders result;
    result.bondOrders = std::move(bondOrders);
    result.diatomicCovalent = std::move(diatomic);
    result.atomicIonicValence = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nAtoms));
    result.atomicCovalentValence = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nAtoms));
    result.totalValence = total;

    for (std::size_t a = 0; a < nAtoms; ++a)
    {
        result.atomicIonicValence(static_cast<Eigen::Index>(a)) = oneCenter[a].ionic;
        result.atomicCovalentValence(static_cast<Eigen::Index>(a)) = oneCenter[a].covalent;
    }

    return result;
}

} // namespace qcx::properties
