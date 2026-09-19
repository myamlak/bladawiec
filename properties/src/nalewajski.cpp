// Nalewajski-Mrozek bond orders: the quadratic
// valence indices of [Nalewajski1996] in the one-determinantal difference
// approach with the frozen superposition-of-isolated-atoms (SAL) reference,
// ported from niedoida's NalewajskiAnalysis (read-only reference). The
// isolated-atom fragment SCFs run internally per distinct element; the
// per-atom valences are averaged over the degenerate-shell configurations
// and combined into the Scheme-III weighted bond orders.
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

// The shell-degeneracy sequence of the reference implementation
// (niedoida's AtomicFragment::AtomicFragment): the shells in Madelung
// (Aufbau, n + l) order - 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p
// 7s 5f 6d 7p - matching how a neutral atom fills. The
// principal-quantum-number order would classify Z <= 18 identically and
// disagree at Z = 19 (Madelung leaves 4s partially filled, n-order would
// leave 3d); naive filling marks the first shell that cannot be fully
// occupied. The reference documents the table's range as "less than 20
// shells" (Z <= 19), exactly the range qcx supports.
inline constexpr std::array<int, 19> kShellDegeneracies{
    1, 1, 3, 1, 3, 1, 5, 3, 1, 5, 3, 1, 7, 5, 3, 1, 7, 5, 3};

// The isolated-atom state classification: how many closed-shell (core)
// states, how many degenerate (partially filled shell) states, and how
// many electrons the degenerate shell holds.
struct FragmentStates {
    int coreStates = 0;
    int degenerateStates = 0;
    int degenerateElectrons = 0;
};

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
// origin in its ground-state multiplicity (the reference's Hund rule:
// multiplicity = n_deg + 1 for a less-than-half-filled shell, else
// 2 n_states - n_deg + 1), run in the MOLECULAR basis set - the same
// single-element-atom construction the SAD guess uses. The fragment
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

// All C(n, k) index combinations (the reference's OccGenerator; the config
// order is irrelevant - the configs are averaged with equal weights).
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

// The one-center valence accumulation of one atom (the reference's
// FragmentConfig loop): the config-averaged v_c_a (off-diagonal) and
// v_i_a (diagonal) of Delta = P1f - P0f over all degenerate-shell
// configurations, P0f the fragment density in the fragment's own Lowdin
// basis (frozen-SAL: each fragment is orthogonalized separately).
struct OneCenterValences {
    double covalent = 0.0; ///< v_c_a.
    double ionic = 0.0; ///< v_i_a.
};

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

    // The reference's occupation logic: a less-than-half-filled degenerate
    // shell holds only alpha electrons (the configs choose which states); a
    // half-or-more filled shell fills alpha completely and varies beta (the
    // configs choose which states the remaining electrons occupy). The
    // non-varying channel contributes its single combination to every
    // config - the configuration count is the varying channel's count.
    const int kAlpha = nDegenerateElectrons <= nDegenerate ? nDegenerateElectrons : nDegenerate;
    const int kBeta = nDegenerateElectrons <= nDegenerate ? 0 : nDegenerateElectrons - nDegenerate;
    const std::vector<std::vector<int>> alphaConfigs = BuildCombinations(nDegenerate, kAlpha);
    const std::vector<std::vector<int>> betaConfigs = BuildCombinations(nDegenerate, kBeta);
    const bool alphaVaries = nDegenerateElectrons <= nDegenerate;
    const std::vector<std::vector<int>>& varyingConfigs = alphaVaries ? alphaConfigs : betaConfigs;
    const std::vector<int>& fixedStates = alphaVaries ? betaConfigs.front() : alphaConfigs.front();
    const std::size_t configCount = varyingConfigs.size();

    // The closed-shell core states are common to every configuration and
    // to both spin channels (in the AO basis; P_core = C_core C_core^T).
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

        const Eigen::MatrixXd p0fAlpha = root * alphaDensity * root;
        const Eigen::MatrixXd p0fBeta = root * betaDensity * root;
        const Eigen::MatrixXd dAlpha = p1fAlpha - p0fAlpha;
        const Eigen::MatrixXd dBeta = p1fBeta - p0fBeta;

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < i; ++j)
            {
                covalent += dAlpha(i, j) * dAlpha(i, j) + dBeta(i, j) * dBeta(i, j);
            }

            ionic += 0.5 * (dAlpha(i, i) * dAlpha(i, i) + dBeta(i, i) * dBeta(i, i));
        }
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

    // The molecular Lowdin basis (the Gopinathan-Jug forward root shared
    // with populations.hpp / eddb.cpp): P1 = S^{1/2} P S^{1/2}.
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

        // The fragment's own Lowdin root uses the atom's overlap block
        // (the frozen-SAL "separately orthogonalized per fragment").
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

    // The interatomic quadratic contributions: the molecular orthogonalized
    // density's off-diagonal blocks (the reference's v_ab; no P0 term - the
    // two functions live on different fragments).
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

    // The interatomic valence (the Scheme-III denominator).
    const Eigen::VectorXd total = diatomic.rowwise().sum();

    // The Scheme-III weighted bond orders (eq. 11 of [Nalewajski1996], the
    // reference's B(a,b)). An atom with no interatomic valence contributes
    // no weighting (its v_ab vanish, so the bond order is zero regardless).
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
