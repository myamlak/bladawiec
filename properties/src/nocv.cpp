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
//
// Four references share the machinery:
//  * closed-shell Hartree-Fock (AnalyzeNocvEts) - the restricted functional
//    at the spin-summed densities, plus its per-spin resolution;
//  * unrestricted Hartree-Fock (AnalyzeNocvEtsUnrestricted) - two
//    independent NOCV problems, each spin with its own transition Fock;
//  * closed-shell Kohn-Sham (AnalyzeNocvEtsKohnSham) - the Ziegler-Rauk
//    three-point transition state, every Fock/energy from the caller's
//    functional seam (the XC potential at all three densities);
//  * unrestricted Kohn-Sham (AnalyzeNocvEtsUnrestrictedKohnSham) - the
//    same three-point state applied per spin, each channel through its own
//    spin's Fock.
// The restricted open-shell reference is refused by name (the file comment
// of nocv.hpp states what the restricted formulation still needs), and
// CheckNocvEtsReference is that refusal's one home.
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

// The functional seam's evaluation, with a named refusal when the callback
// is missing: a Kohn-Sham decomposition cannot run without it, and an empty
// std::function would otherwise throw.
qcx::Result<qcx::grid::XcEvaluation> EvaluateFunctional(const NocvEtsFunctional& functional,
                                                        const Eigen::MatrixXd& densityAlpha,
                                                        const Eigen::MatrixXd& densityBeta) {
    if (!functional.evaluate)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the Kohn-Sham ETS-NOCV needs a callable XC "
                                          "evaluator seam (NocvEtsFunctional::evaluate)"});
    }

    return functional.evaluate(densityAlpha, densityBeta);
}

// The closed-shell Kohn-Sham Fock matrix and total energy at an arbitrary
// spin-summed density rho:
//
//     F = c_HF F_HF[rho] + (1 - c_HF) (H + J[rho])
//         + (V_alpha + V_beta) / 2,
//     E = c_HF E_HF[rho] + (1 - c_HF) E_coul[rho]
//         + E_xc[rho/2, rho/2],
//
// where F_HF = H + J - K/2 and E_HF/E_coul are the RHF evaluator's totals.
// The weighted Fock IS H + J - c_HF K/2 + V_xc (F_HF carries K/2 and the
// Coulomb-only Fock carries H + J), and the weighted energy keeps the
// nuclear repulsion once because both totals carry it. The XC evaluation is
// the engine's closed-shell convention: the spin split is rho/2 per spin and
// the restricted potential is the HALVED sum of the spin potentials.
struct KsClosedShellEvaluation {
    Eigen::MatrixXd fock; ///< The closed-shell Kohn-Sham Fock at rho.
    double totalEnergy = 0.0; ///< The Kohn-Sham total energy at rho (Hartree).
};

qcx::Result<KsClosedShellEvaluation> EvaluateKsClosedShell(
    const qcx::molecule::Molecule& molecule,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& density,
    const NocvEtsFunctional& functional) {
    auto full = qcx::scf::EvaluateRhfDensityEnergy(molecule, coreHamiltonian, eri, density);

    if (!full.has_value())
    {
        return std::unexpected(full.error());
    }

    auto coulomb = qcx::scf::EvaluateRhfDensityEnergy(
        molecule, coreHamiltonian, eri, density, /*includeExchange=*/false);

    if (!coulomb.has_value())
    {
        return std::unexpected(coulomb.error());
    }

    auto xc = EvaluateFunctional(functional, 0.5 * density, 0.5 * density);

    if (!xc.has_value())
    {
        return std::unexpected(xc.error());
    }

    const double fraction = functional.exchangeFraction;
    KsClosedShellEvaluation result;
    result.fock = fraction * full->fock + (1.0 - fraction) * coulomb->fock +
                  0.5 * (xc->potentialAlpha + xc->potentialBeta);
    result.totalEnergy =
        fraction * full->totalEnergy + (1.0 - fraction) * coulomb->totalEnergy + xc->energy;
    return result;
}

// The per-spin Kohn-Sham Focks and the pieces the UHF loop's energy seam
// needs at one per-spin density pair:
//
//     F_sigma = c_HF F_sigma^HF + (1 - c_HF) (H + J[D_total]) + V_sigma,
//     contribution = E_xc - c_HF/2 sum_sigma Tr[D_sigma K[D_sigma]],
//
// with K[D_sigma] = (H + J) - F_sigma^HF recovered from the two evaluator
// calls. This is the fragment SCF's composition: the fragments may be
// open-shell (the Hund convention of AnalyzeNocvEts), so the functional is a
// functional of the spin pair, not of the total density.
struct KsSpinEvaluation {
    Eigen::MatrixXd fockAlpha; ///< F_alpha at (D_a, D_b).
    Eigen::MatrixXd fockBeta; ///< F_beta at (D_a, D_b).
    Eigen::MatrixXd exchangeAlpha; ///< K[D_a].
    Eigen::MatrixXd exchangeBeta; ///< K[D_b].
    double xcEnergy = 0.0; ///< E_xc(D_a, D_b).
    /// The Kohn-Sham total energy at (D_a, D_b), the same weighted total
    /// the fragment SCF's seam composes (nuclear repulsion counted once).
    double totalEnergy = 0.0;
};

qcx::Result<KsSpinEvaluation> EvaluateKsSpin(
    const qcx::molecule::Molecule& molecule,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    const NocvEtsFunctional& functional) {
    auto full = qcx::scf::EvaluateUhfDensityEnergy(
        molecule, coreHamiltonian, eri, densityAlpha, densityBeta);

    if (!full.has_value())
    {
        return std::unexpected(full.error());
    }

    auto coulomb = qcx::scf::EvaluateUhfDensityEnergy(
        molecule, coreHamiltonian, eri, densityAlpha, densityBeta, /*includeExchange=*/false);

    if (!coulomb.has_value())
    {
        return std::unexpected(coulomb.error());
    }

    auto xc = EvaluateFunctional(functional, densityAlpha, densityBeta);

    if (!xc.has_value())
    {
        return std::unexpected(xc.error());
    }

    const double fraction = functional.exchangeFraction;
    KsSpinEvaluation result;
    result.exchangeAlpha = coulomb->fockAlpha - full->fockAlpha;
    result.exchangeBeta = coulomb->fockBeta - full->fockBeta;
    result.fockAlpha =
        fraction * full->fockAlpha + (1.0 - fraction) * coulomb->fockAlpha + xc->potentialAlpha;
    result.fockBeta =
        fraction * full->fockBeta + (1.0 - fraction) * coulomb->fockBeta + xc->potentialBeta;
    result.xcEnergy = xc->energy;
    // The same weighted total the fragment SCF's energy seam forms:
    // E = c_HF E_HF + (1 - c_HF) E_coul + E_xc, and both evaluator totals
    // carry the nuclear repulsion once, so the weighted sum counts it once.
    result.totalEnergy =
        fraction * full->totalEnergy + (1.0 - fraction) * coulomb->totalEnergy + xc->energy;
    return result;
}

// The ROHF/ROKS refusal: the restricted formulation is the missing piece,
// and the message names it rather than the path's absence.
qcx::Error RestrictedOpenShellRefusal() {
    return qcx::Error{
        qcx::ErrorCode::kUnimplemented,
        "the restricted open-shell ETS-NOCV (ROHF/ROKS) is not served: a restricted "
        "open-shell determinant's two spin densities are each idempotent, so the "
        "unrestricted path would run on it unchanged and return NOCV pairs of different "
        "spatial shapes - which the restricted reference, ONE spatial orbital set, does "
        "not permit. The ROHF/ROKS energy is also not a functional of the total density "
        "(it depends on the closed/open-shell split), so the restricted formulation needs "
        "a transition-state Fock defined from the ROHF effective Fock operators (F_c and "
        "F_o) and a channel pairing rule for a spectrum that is not +-nu paired. Neither "
        "exists in this module; use the unrestricted or closed-shell entry point, or "
        "implement the restricted transition-state Fock first"};
}

// One fragment's ETS embedding: its own SCF at its molecular geometry in
// the full molecular basis, plus the spin-summed frozen density, the
// per-spin occupied blocks that span the combined occupied space, and the
// fragment's own one-electron operators (its nuclear repulsion and core
// Hamiltonian feed the Coulomb-only fragment self-energy).
struct EtsFragment {
    double energy = 0.0; ///< The fragment SCF total energy E_frag_i.
    Eigen::MatrixXd density; ///< P_i = D_a + D_b (n x n, spin-summed).
    Eigen::MatrixXd densityAlpha; ///< D_a = C_a C_a^T (1x convention).
    Eigen::MatrixXd densityBeta; ///< D_b = C_b C_b^T (1x convention).
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
    const qcx::scf::UhfOptions& options,
    const NocvEtsFunctional* functional = nullptr) {
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
    // fragment orbitals span the whole space), the fragment's own core. A
    // Kohn-Sham decomposition hands the loop the functional seam instead of
    // the dense supermatrix Fock (the same SCF loop, the same convergence
    // gate); the Hartree-Fock reference keeps the supermatrix call unchanged.
    qcx::Result<qcx::scf::UhfResult> scf =
        functional == nullptr
            ? qcx::scf::RunUhfScf(*fragment, overlap, core, eri, options)
            : qcx::scf::RunUhfScf(
                  *fragment,
                  overlap,
                  core,
                  options,
                  [&fragment, &core, &eri, functional](const Eigen::MatrixXd& densityAlpha,
                                                       const Eigen::MatrixXd& densityBeta)
                      -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
                      auto spin = EvaluateKsSpin(
                          *fragment, core, eri, densityAlpha, densityBeta, *functional);

                      if (!spin.has_value())
                      {
                          return std::unexpected(spin.error());
                      }

                      return std::make_pair(std::move(spin->fockAlpha), std::move(spin->fockBeta));
                  },
                  [&fragment, &core, &eri](
                      const Eigen::MatrixXd& totalDensity) -> qcx::Result<Eigen::MatrixXd> {
                      const Eigen::MatrixXd zero =
                          Eigen::MatrixXd::Zero(totalDensity.rows(), totalDensity.cols());
                      auto coulomb = qcx::scf::EvaluateUhfDensityEnergy(
                          *fragment, core, eri, totalDensity, zero, /*includeExchange=*/false);

                      if (!coulomb.has_value())
                      {
                          return std::unexpected(coulomb.error());
                      }

                      return Eigen::MatrixXd(coulomb->fockAlpha - core);
                  },
                  [&fragment, &core, &eri, functional](
                      const Eigen::MatrixXd& densityAlpha,
                      const Eigen::MatrixXd& densityBeta) -> qcx::Result<double> {
                      auto spin = EvaluateKsSpin(
                          *fragment, core, eri, densityAlpha, densityBeta, *functional);

                      if (!spin.has_value())
                      {
                          return std::unexpected(spin.error());
                      }

                      const double exchangeTrace =
                          (densityAlpha.cwiseProduct(spin->exchangeAlpha)).sum() +
                          (densityBeta.cwiseProduct(spin->exchangeBeta)).sum();
                      return spin->xcEnergy - 0.5 * functional->exchangeFraction * exchangeTrace;
                  });

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
                       scf->densityAlpha,
                       scf->densityBeta,
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
    Eigen::VectorXd eigenvalues; ///< The +nu_k of each channel (electrons).
    /// The -nu_k partner of each channel; the unpaired tail channel of an
    /// odd function count reports 0 (it has no partner), and a rank-imbalance
    /// leftover pairs with a ~0 eigenvalue.
    Eigen::VectorXd negativeEigenvalues;
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
    set.negativeEigenvalues.resize(static_cast<Eigen::Index>(channelCount));

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
        set.negativeEigenvalues(static_cast<Eigen::Index>(pair)) = lambdaMinus;
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
        set.negativeEigenvalues(static_cast<Eigen::Index>(pairCount)) = 0.0;
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

// The matrix is n x n and the ERI tensor is {n, n, n, n}: the one shape
// check every entry point runs before any contraction.
qcx::Result<std::size_t> CheckNocvShapes(
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& coreHamiltonian,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd* densityBeta) {
    const std::size_t n = eri.Shape()[0];

    if (eri.Shape()[1] != n || eri.Shape()[2] != n || eri.Shape()[3] != n ||
        static_cast<std::size_t>(coreHamiltonian.rows()) != n ||
        static_cast<std::size_t>(coreHamiltonian.cols()) != n ||
        static_cast<std::size_t>(densityAlpha.rows()) != n ||
        static_cast<std::size_t>(densityAlpha.cols()) != n ||
        (densityBeta != nullptr && (static_cast<std::size_t>(densityBeta->rows()) != n ||
                                    static_cast<std::size_t>(densityBeta->cols()) != n)))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "core Hamiltonian and density must match the ERI function count"});
    }

    return n;
}

// The fragment partition must be disjoint and cover every atom once.
qcx::Result<void> ValidateFragmentPartition(
    const qcx::molecule::Molecule& molecule,
    const std::vector<std::vector<std::size_t>>& fragments) {
    const std::size_t atomCount = molecule.AtomCount();
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

    return {};
}

// The frozen fragment set: every fragment's SCF record, the summed frozen
// density, and the two energy sums the ETS partition needs.
struct FrozenFragmentSet {
    std::vector<EtsFragment> scfs;
    Eigen::MatrixXd density; ///< P_frag = sum_i (D_a,i + D_b,i).
    double energySum = 0.0; ///< sum_i E_frag,i.
    double coulombEnergySum = 0.0; ///< sum_i E_coul-fragment[D_i] (with E_nuc,i).
};

// Runs every fragment SCF and accumulates the frozen set. The Coulomb-only
// fragment self-energy is evaluated at the fragment's own molecule and core
// (its own nuclei), so summing it cancels the fragment nuclear repulsions
// and leaves the classical cross-fragment interaction in E_elstat.
qcx::Result<FrozenFragmentSet> RunFragmentSet(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& overlap,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const std::vector<std::vector<std::size_t>>& fragments,
    const qcx::scf::UhfOptions& fragmentOptions,
    const NocvEtsFunctional* functional,
    Eigen::VectorXd& fragmentEnergies) {
    FrozenFragmentSet set;
    set.density = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(eri.Shape()[0]),
                                        static_cast<Eigen::Index>(eri.Shape()[0]));
    set.scfs.reserve(fragments.size());

    for (std::size_t i = 0; i < fragments.size(); ++i)
    {
        auto scf = RunEtsFragmentScf(
            molecule, basisSet, overlap, eri, fragments[i], fragmentOptions, functional);

        if (!scf.has_value())
        {
            return std::unexpected(scf.error());
        }

        fragmentEnergies(static_cast<Eigen::Index>(i)) = scf->energy;
        set.energySum += scf->energy;
        set.density += scf->density;

        auto coulomb = qcx::scf::EvaluateRhfDensityEnergy(
            scf->molecule, scf->coreHamiltonian, eri, scf->density, /*includeExchange=*/false);

        if (!coulomb.has_value())
        {
            return std::unexpected(coulomb.error());
        }

        set.coulombEnergySum += coulomb->totalEnergy;
        set.scfs.push_back(std::move(*scf));
    }

    return set;
}

// The per-spin projectors onto the combined fragment occupied spaces:
// C_sigma (C_sigma^T S C_sigma)^-1 C_sigma^T with NO per-channel factor (the
// 2 P convention appears only as the SUM of two equal closed-shell
// channels; a per-channel 2 would double the open-shell density difference
// and break the H2/STO-3G limit).
struct OrthogonalizedChannels {
    Eigen::MatrixXd alpha;
    Eigen::MatrixXd beta;
};

OrthogonalizedChannels BuildOrthogonalizedChannels(const std::vector<EtsFragment>& scfs,
                                                   const Eigen::MatrixXd& overlap,
                                                   std::size_t n) {
    std::vector<Eigen::MatrixXd> alphaBlocks;
    std::vector<Eigen::MatrixXd> betaBlocks;
    alphaBlocks.reserve(scfs.size());
    betaBlocks.reserve(scfs.size());

    for (const auto& scf : scfs)
    {
        alphaBlocks.push_back(scf.occupiedAlpha);
        betaBlocks.push_back(scf.occupiedBeta);
    }

    const Eigen::MatrixXd combinedAlpha = JoinBlocks(alphaBlocks, static_cast<Eigen::Index>(n));
    const Eigen::MatrixXd combinedBeta = JoinBlocks(betaBlocks, static_cast<Eigen::Index>(n));
    const Eigen::MatrixXd overlapAlpha = combinedAlpha.transpose() * overlap * combinedAlpha;
    const Eigen::MatrixXd overlapBeta = combinedBeta.transpose() * overlap * combinedBeta;

    OrthogonalizedChannels channels;
    channels.alpha = combinedAlpha * overlapAlpha.inverse() * combinedAlpha.transpose();
    channels.beta = combinedBeta * overlapBeta.inverse() * combinedBeta.transpose();
    return channels;
}

// The Lowdin transform pair the channel resolution runs in: root is S^(1/2)
// and rootInverse its inverse (the same matrix, from one eigendecomposition).
struct LowdinTransform {
    Eigen::MatrixXd root;
    Eigen::MatrixXd rootInverse;
};

LowdinTransform BuildLowdinTransform(const Eigen::MatrixXd& overlap) {
    LowdinTransform transform;
    transform.root = SymmetricSquareRootOfOverlap(overlap);
    transform.rootInverse = transform.root.inverse();
    return transform;
}

} // namespace

qcx::Result<void> CheckNocvEtsReference(NocvEtsReference reference, NocvEtsEntryPoint entryPoint) {
    // The restricted open-shell reference is refused before the entry point
    // is consulted: no entry point serves it, and its statement is the same
    // whichever lane the caller intended.
    if (reference == NocvEtsReference::kRestrictedOpenShell)
    {
        return std::unexpected(RestrictedOpenShellRefusal());
    }

    const bool unrestrictedReference = reference == NocvEtsReference::kUnrestricted;
    const bool unrestrictedEntryPoint = entryPoint == NocvEtsEntryPoint::kUnrestrictedHartreeFock ||
                                        entryPoint == NocvEtsEntryPoint::kUnrestrictedKohnSham;

    // The served pairs: the closed-shell reference on the two closed-shell
    // entry points, the unrestricted reference on the two unrestricted ones.
    // A mismatched pair names the entry point that serves the declared
    // reference, so a caller holding the wrong one is redirected rather than
    // left with a generic domain error.
    if (unrestrictedReference != unrestrictedEntryPoint)
    {
        switch (entryPoint)
        {
        case NocvEtsEntryPoint::kClosedShellHartreeFock:
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "AnalyzeNocvEts serves the closed-shell reference: a per-spin "
                           "density pair is AnalyzeNocvEtsUnrestricted's input (the "
                           "unrestricted Hartree-Fock decomposition, one NOCV problem per "
                           "spin)"});

        case NocvEtsEntryPoint::kUnrestrictedHartreeFock:
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "AnalyzeNocvEtsUnrestricted serves the unrestricted reference: a "
                           "closed-shell density pair is AnalyzeNocvEts's input, which "
                           "reports the restricted functional's E_Pauli/E_orb split this "
                           "entry does not compute"});

        case NocvEtsEntryPoint::kClosedShellKohnSham:
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "AnalyzeNocvEtsKohnSham serves the closed-shell reference: a "
                           "per-spin (UKS) density pair is AnalyzeNocvEtsUnrestrictedKohnSham's "
                           "input (the same three-point transition state applied per spin)"});

        case NocvEtsEntryPoint::kUnrestrictedKohnSham:
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "AnalyzeNocvEtsUnrestrictedKohnSham serves the unrestricted "
                           "reference: a spin-summed closed-shell density is "
                           "AnalyzeNocvEtsKohnSham's input (the closed-shell Ziegler-Rauk "
                           "three-point transition state)"});
        }

        // Unreachable while the enumerator set above is total; the tail keeps
        // the mismatch fail-closed if a new entry point is added without a
        // case, and the switch stays total without a `default:` arm that
        // would hide that new entry point.
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "unclassified ETS-NOCV entry point"});
    }

    // The reference and the entry point agree: the two closed-shell reference
    // forms on the closed-shell entry points, the two unrestricted forms on
    // the unrestricted ones.
    return {};
}

qcx::Result<NocvEtsDecomposition> AnalyzeNocvEts(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& density,
    const std::vector<std::vector<std::size_t>>& fragments,
    const qcx::scf::UhfOptions& fragmentOptions) {
    auto functionCount = CheckNocvShapes(eri, coreHamiltonian, density, nullptr);

    if (!functionCount.has_value())
    {
        return std::unexpected(functionCount.error());
    }

    const std::size_t n = *functionCount;
    auto partition = ValidateFragmentPartition(molecule, fragments);

    if (!partition.has_value())
    {
        return std::unexpected(partition.error());
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
    auto frozen = RunFragmentSet(molecule,
                                 basisSet,
                                 overlapEigen,
                                 eri,
                                 fragments,
                                 fragmentOptions,
                                 /*functional=*/nullptr,
                                 result.fragmentEnergies);

    if (!frozen.has_value())
    {
        return std::unexpected(frozen.error());
    }

    result.fragmentDensity = frozen->density;

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

    result.electrostatic = combinedCoulomb->totalEnergy - frozen->coulombEnergySum;

    // P_orth: the per-spin-channel projector onto the combined occupied
    // space, C_occ (C_occ^T S C_occ)^-1 C_occ^T with NO per-channel factor.
    // The spin-summed density convention of 2 P appears only as the SUM of
    // two equal closed-shell channels (2 P = P_a + P_b when P_a = P_b); a
    // per-channel 2 would double the open-shell density difference and
    // break the H2/STO-3G limit (two doublets spanning the full space give
    // P_orth = 2 S^-1 = D_mol and E_orb = 0 exactly).
    const auto projectors = BuildOrthogonalizedChannels(frozen->scfs, overlapEigen, n);
    result.orthogonalizedDensityAlpha = projectors.alpha;
    result.orthogonalizedDensityBeta = projectors.beta;
    result.orthogonalizedDensity = projectors.alpha + projectors.beta;

    // E_Pauli: what the antisymmetrized, orthogonalized fragment
    // determinant adds beyond the frozen combination's classical
    // interaction (defined so the partition identity holds exactly).
    auto orthEnergy = qcx::scf::EvaluateRhfDensityEnergy(
        molecule, coreHamiltonian, eri, result.orthogonalizedDensity);

    if (!orthEnergy.has_value())
    {
        return std::unexpected(orthEnergy.error());
    }

    result.pauli = orthEnergy->totalEnergy - frozen->energySum - result.electrostatic;

    // The NOCV channels of Delta P = P_mol - P_orth: the EXACT resolution
    // E[P_mol] - E[P_orth] = Tr[Delta P F^TS] of the quadratic RHF
    // functional (E[P1] - E[P2] = Tr[(P1 - P2) F^TS] with the
    // transition-state Fock matrix F^TS evaluated at 1/2(P1 + P2), by the
    // symmetry of G = J - 1/2 K). Every eigenpair of the Lowdin
    // orthogonalized difference contributes its own diagonal element.
    const Eigen::MatrixXd deltaP = density - result.orthogonalizedDensity;
    const LowdinTransform lowdin = BuildLowdinTransform(overlapEigen);
    const Eigen::MatrixXd deltaPLowdin = lowdin.root * deltaP * lowdin.root;

    auto transitionFock = qcx::scf::EvaluateRhfDensityEnergy(
        molecule, coreHamiltonian, eri, 0.5 * (result.orthogonalizedDensity + density));

    if (!transitionFock.has_value())
    {
        return std::unexpected(transitionFock.error());
    }

    // The transition-state Fock matrix in the Lowdin basis, where the
    // channel diagonal elements read v^T F^TS v directly.
    const Eigen::MatrixXd fockLowdin =
        lowdin.rootInverse * transitionFock->fock * lowdin.rootInverse;
    const auto channels = ResolveChannels(deltaPLowdin, fockLowdin);
    result.orbitalComponents = channels.components;
    result.nocvEigenvalues = channels.eigenvalues;
    result.nocvEigenvaluesNegative = channels.negativeEigenvalues;
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
        lowdin.rootInverse * transitionUf->fockAlpha * lowdin.rootInverse;
    const Eigen::MatrixXd fockBetaLowdin =
        lowdin.rootInverse * transitionUf->fockBeta * lowdin.rootInverse;
    const auto alphaChannels =
        ResolveChannels(lowdin.root * deltaAlpha * lowdin.root, fockAlphaLowdin);
    const auto betaChannels =
        ResolveChannels(lowdin.root * deltaBeta * lowdin.root, fockBetaLowdin);
    result.orbitalComponentsAlpha = alphaChannels.components;
    result.nocvEigenvaluesAlpha = alphaChannels.eigenvalues;
    result.nocvEigenvaluesAlphaNegative = alphaChannels.negativeEigenvalues;
    result.orbitalComponentsBeta = betaChannels.components;
    result.nocvEigenvaluesBeta = betaChannels.eigenvalues;
    result.nocvEigenvaluesBetaNegative = betaChannels.negativeEigenvalues;
    result.orbitalUnrestricted = alphaChannels.energy + betaChannels.energy;

    // The deformation densities: the closed-shell reference splits its
    // molecular density as D/2 per spin, the same convention the spin
    // resolution above uses.
    result.deformationDensity = deltaP;
    result.deformationDensityAlpha = deltaAlpha;
    result.deformationDensityBeta = deltaBeta;

    result.bindingEnergy = result.electrostatic + result.pauli + result.orbital;
    return result;
}

qcx::Result<NocvEtsDecomposition> AnalyzeNocvEtsUnrestricted(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    NocvEtsReference reference,
    const std::vector<std::vector<std::size_t>>& fragments,
    const qcx::scf::UhfOptions& fragmentOptions) {
    auto referenceGate =
        CheckNocvEtsReference(reference, NocvEtsEntryPoint::kUnrestrictedHartreeFock);

    if (!referenceGate.has_value())
    {
        return std::unexpected(referenceGate.error());
    }

    auto functionCount = CheckNocvShapes(eri, coreHamiltonian, densityAlpha, &densityBeta);

    if (!functionCount.has_value())
    {
        return std::unexpected(functionCount.error());
    }

    const std::size_t n = *functionCount;
    auto partition = ValidateFragmentPartition(molecule, fragments);

    if (!partition.has_value())
    {
        return std::unexpected(partition.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    const Eigen::MatrixXd overlapEigen = internal::TensorToEigen(*overlap);

    // The fragments run the same UHF SCFs as the closed-shell entry point's
    // (the fragment spin states are a property of the fragment, not of the
    // molecular reference).
    NocvEtsDecomposition result;
    result.fragmentEnergies.resize(static_cast<Eigen::Index>(fragments.size()));
    auto frozen = RunFragmentSet(molecule,
                                 basisSet,
                                 overlapEigen,
                                 eri,
                                 fragments,
                                 fragmentOptions,
                                 /*functional=*/nullptr,
                                 result.fragmentEnergies);

    if (!frozen.has_value())
    {
        return std::unexpected(frozen.error());
    }

    result.fragmentDensity = frozen->density;

    // E_elstat: the same Coulomb-only cross-fragment interaction the
    // closed-shell entry point forms (a functional-independent quantity).
    auto combinedCoulomb = qcx::scf::EvaluateRhfDensityEnergy(
        molecule, coreHamiltonian, eri, result.fragmentDensity, /*includeExchange=*/false);

    if (!combinedCoulomb.has_value())
    {
        return std::unexpected(combinedCoulomb.error());
    }

    result.electrostatic = combinedCoulomb->totalEnergy - frozen->coulombEnergySum;

    const auto projectors = BuildOrthogonalizedChannels(frozen->scfs, overlapEigen, n);
    result.orthogonalizedDensityAlpha = projectors.alpha;
    result.orthogonalizedDensityBeta = projectors.beta;
    result.orthogonalizedDensity = projectors.alpha + projectors.beta;

    // E_Pauli of the UHF functional: the orthogonalized combined
    // determinant's energy minus the fragments' own, beyond E_elstat. The
    // per-spin densities carry the fragment spin polarization exactly, so
    // the identity E_UHF[P_mol] - sum E_frag = E_elstat + E_Pauli + E_orb
    // holds with the UHF functional throughout.
    auto orthEnergy = qcx::scf::EvaluateUhfDensityEnergy(
        molecule, coreHamiltonian, eri, projectors.alpha, projectors.beta);

    if (!orthEnergy.has_value())
    {
        return std::unexpected(orthEnergy.error());
    }

    result.pauli = orthEnergy->totalEnergy - frozen->energySum - result.electrostatic;

    // The two INDEPENDENT NOCV problems. Each spin density of a UHF
    // determinant is a projector, so Delta P_sigma = D_sigma - P_orth,sigma
    // is a difference of two projectors and keeps the sign-paired (+-nu_k)
    // spectrum. The transition Fock of each spin is that spin's own
    // F_sigma^TS = h + J[P^TS_total] - K[D_sigma^TS] at
    // D_sigma^TS = 1/2(P_orth,sigma + D_sigma); the midpoint is exact for
    // the quadratic UHF functional, so the per-spin channel sums ARE the
    // total orbital relaxation.
    const LowdinTransform lowdin = BuildLowdinTransform(overlapEigen);
    const Eigen::MatrixXd deltaAlpha = densityAlpha - projectors.alpha;
    const Eigen::MatrixXd deltaBeta = densityBeta - projectors.beta;
    const Eigen::MatrixXd transitionAlpha = 0.5 * (projectors.alpha + densityAlpha);
    const Eigen::MatrixXd transitionBeta = 0.5 * (projectors.beta + densityBeta);

    auto transitionFock = qcx::scf::EvaluateUhfDensityEnergy(
        molecule, coreHamiltonian, eri, transitionAlpha, transitionBeta);

    if (!transitionFock.has_value())
    {
        return std::unexpected(transitionFock.error());
    }

    const Eigen::MatrixXd fockAlphaLowdin =
        lowdin.rootInverse * transitionFock->fockAlpha * lowdin.rootInverse;
    const Eigen::MatrixXd fockBetaLowdin =
        lowdin.rootInverse * transitionFock->fockBeta * lowdin.rootInverse;
    const auto alphaChannels =
        ResolveChannels(lowdin.root * deltaAlpha * lowdin.root, fockAlphaLowdin);
    const auto betaChannels =
        ResolveChannels(lowdin.root * deltaBeta * lowdin.root, fockBetaLowdin);
    result.orbitalComponentsAlpha = alphaChannels.components;
    result.nocvEigenvaluesAlpha = alphaChannels.eigenvalues;
    result.nocvEigenvaluesAlphaNegative = alphaChannels.negativeEigenvalues;
    result.orbitalComponentsBeta = betaChannels.components;
    result.nocvEigenvaluesBeta = betaChannels.eigenvalues;
    result.nocvEigenvaluesBetaNegative = betaChannels.negativeEigenvalues;
    result.orbitalUnrestricted = alphaChannels.energy + betaChannels.energy;

    // The total deformation density is the sum of the two spin problems.
    result.deformationDensityAlpha = deltaAlpha;
    result.deformationDensityBeta = deltaBeta;
    result.deformationDensity = deltaAlpha + deltaBeta;

    // The unrestricted reference's E_orb is the spin-resolved one; there is
    // no restricted functional on a UHF determinant to report in parallel.
    result.orbital = result.orbitalUnrestricted;
    result.bindingEnergy = result.electrostatic + result.pauli + result.orbital;
    return result;
}

qcx::Result<NocvEtsDecomposition> AnalyzeNocvEtsKohnSham(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& density,
    NocvEtsReference reference,
    const NocvEtsFunctional& functional,
    const std::vector<std::vector<std::size_t>>& fragments,
    const qcx::scf::UhfOptions& fragmentOptions) {
    auto referenceGate = CheckNocvEtsReference(reference, NocvEtsEntryPoint::kClosedShellKohnSham);

    if (!referenceGate.has_value())
    {
        return std::unexpected(referenceGate.error());
    }

    if (!functional.evaluate)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the Kohn-Sham ETS-NOCV needs a callable XC "
                                          "evaluator seam (NocvEtsFunctional::evaluate)"});
    }

    if (!(functional.exchangeFraction >= 0.0 && functional.exchangeFraction <= 1.0))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the exact-exchange fraction must lie in [0, 1]"});
    }

    auto functionCount = CheckNocvShapes(eri, coreHamiltonian, density, nullptr);

    if (!functionCount.has_value())
    {
        return std::unexpected(functionCount.error());
    }

    const std::size_t n = *functionCount;
    auto partition = ValidateFragmentPartition(molecule, fragments);

    if (!partition.has_value())
    {
        return std::unexpected(partition.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    const Eigen::MatrixXd overlapEigen = internal::TensorToEigen(*overlap);

    // The fragments run their own SCFs through the functional seam (the
    // same UHF loop, with the Kohn-Sham Fock and energy callbacks), so their
    // energies are the same functional's - open-shell fragments (the Hund
    // convention) come out as UKS fragments.
    NocvEtsDecomposition result;
    result.fragmentEnergies.resize(static_cast<Eigen::Index>(fragments.size()));
    auto frozen = RunFragmentSet(molecule,
                                 basisSet,
                                 overlapEigen,
                                 eri,
                                 fragments,
                                 fragmentOptions,
                                 &functional,
                                 result.fragmentEnergies);

    if (!frozen.has_value())
    {
        return std::unexpected(frozen.error());
    }

    result.fragmentDensity = frozen->density;

    auto combinedCoulomb = qcx::scf::EvaluateRhfDensityEnergy(
        molecule, coreHamiltonian, eri, result.fragmentDensity, /*includeExchange=*/false);

    if (!combinedCoulomb.has_value())
    {
        return std::unexpected(combinedCoulomb.error());
    }

    result.electrostatic = combinedCoulomb->totalEnergy - frozen->coulombEnergySum;

    const auto projectors = BuildOrthogonalizedChannels(frozen->scfs, overlapEigen, n);
    result.orthogonalizedDensityAlpha = projectors.alpha;
    result.orthogonalizedDensityBeta = projectors.beta;
    result.orthogonalizedDensity = projectors.alpha + projectors.beta;

    // E_Pauli of the Kohn-Sham functional at the orthogonalized density:
    // the kinetic, XC and exchange energy the antisymmetrized combination
    // adds beyond the frozen classical interaction (defined so the
    // partition identity holds).
    auto orthEnergy = EvaluateKsClosedShell(
        molecule, coreHamiltonian, eri, result.orthogonalizedDensity, functional);

    if (!orthEnergy.has_value())
    {
        return std::unexpected(orthEnergy.error());
    }

    result.pauli = orthEnergy->totalEnergy - frozen->energySum - result.electrostatic;

    // The three-point (Simpson) transition state. With
    // rho_lambda = rho_orth + lambda Delta rho,
    //     E[rho_mol] - E[rho_orth] = integral_0^1 Tr[Delta rho F[rho_lambda]] dlambda,
    // and Simpson's rule replaces the integrand by
    //     F^TS = 1/6 F[rho_orth] + 2/3 F[rho_TS] + 1/6 F[rho_mol].
    // The XC potential is evaluated at each of the three densities, so a
    // non-linear functional enters all three samples; for a quadratic
    // functional the weighted average collapses to F[rho_TS] exactly.
    const Eigen::MatrixXd& densityOrth = result.orthogonalizedDensity;
    const Eigen::MatrixXd densityTransition = 0.5 * (densityOrth + density);

    auto fockOrth = EvaluateKsClosedShell(molecule, coreHamiltonian, eri, densityOrth, functional);

    if (!fockOrth.has_value())
    {
        return std::unexpected(fockOrth.error());
    }

    auto fockTransition =
        EvaluateKsClosedShell(molecule, coreHamiltonian, eri, densityTransition, functional);

    if (!fockTransition.has_value())
    {
        return std::unexpected(fockTransition.error());
    }

    auto fockMolecular = EvaluateKsClosedShell(molecule, coreHamiltonian, eri, density, functional);

    if (!fockMolecular.has_value())
    {
        return std::unexpected(fockMolecular.error());
    }

    const Eigen::MatrixXd transitionStateFock = (1.0 / 6.0) * fockOrth->fock +
                                                (2.0 / 3.0) * fockTransition->fock +
                                                (1.0 / 6.0) * fockMolecular->fock;

    const LowdinTransform lowdin = BuildLowdinTransform(overlapEigen);
    const Eigen::MatrixXd deltaP = density - densityOrth;
    const Eigen::MatrixXd fockLowdin =
        lowdin.rootInverse * transitionStateFock * lowdin.rootInverse;
    const auto channels = ResolveChannels(lowdin.root * deltaP * lowdin.root, fockLowdin);
    result.orbitalComponents = channels.components;
    result.nocvEigenvalues = channels.eigenvalues;
    result.nocvEigenvaluesNegative = channels.negativeEigenvalues;
    result.orbital = channels.energy;

    // The closed-shell reference splits the molecular density as D/2 per
    // spin, the same convention its per-spin resolution uses; the Kohn-Sham
    // functional itself is spin-summed, so orbitalUnrestricted carries the
    // same value rather than a second resolution.
    result.deformationDensity = deltaP;
    result.deformationDensityAlpha = 0.5 * deltaP;
    result.deformationDensityBeta = 0.5 * deltaP;
    result.orbitalUnrestricted = result.orbital;

    result.bindingEnergy = result.electrostatic + result.pauli + result.orbital;
    return result;
}

qcx::Result<NocvEtsDecomposition> AnalyzeNocvEtsUnrestrictedKohnSham(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    NocvEtsReference reference,
    const NocvEtsFunctional& functional,
    const std::vector<std::vector<std::size_t>>& fragments,
    const qcx::scf::UhfOptions& fragmentOptions) {
    auto referenceGate = CheckNocvEtsReference(reference, NocvEtsEntryPoint::kUnrestrictedKohnSham);

    if (!referenceGate.has_value())
    {
        return std::unexpected(referenceGate.error());
    }

    if (!functional.evaluate)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the Kohn-Sham ETS-NOCV needs a callable XC "
                                          "evaluator seam (NocvEtsFunctional::evaluate)"});
    }

    if (!(functional.exchangeFraction >= 0.0 && functional.exchangeFraction <= 1.0))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the exact-exchange fraction must lie in [0, 1]"});
    }

    auto functionCount = CheckNocvShapes(eri, coreHamiltonian, densityAlpha, &densityBeta);

    if (!functionCount.has_value())
    {
        return std::unexpected(functionCount.error());
    }

    const std::size_t n = *functionCount;
    auto partition = ValidateFragmentPartition(molecule, fragments);

    if (!partition.has_value())
    {
        return std::unexpected(partition.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    const Eigen::MatrixXd overlapEigen = internal::TensorToEigen(*overlap);

    // The fragments run their own SCFs through the functional seam (the same
    // UHF loop and the same convergence gate as every other entry), so an
    // open-shell fragment - the Hund convention - comes out as a UKS
    // fragment.
    NocvEtsDecomposition result;
    result.fragmentEnergies.resize(static_cast<Eigen::Index>(fragments.size()));
    auto frozen = RunFragmentSet(molecule,
                                 basisSet,
                                 overlapEigen,
                                 eri,
                                 fragments,
                                 fragmentOptions,
                                 &functional,
                                 result.fragmentEnergies);

    if (!frozen.has_value())
    {
        return std::unexpected(frozen.error());
    }

    result.fragmentDensity = frozen->density;

    // E_elstat is the functional-independent Coulomb-only cross-fragment
    // interaction every entry forms: the exchange-free functional value of
    // the combined density minus the fragments' own.
    auto combinedCoulomb = qcx::scf::EvaluateRhfDensityEnergy(
        molecule, coreHamiltonian, eri, result.fragmentDensity, /*includeExchange=*/false);

    if (!combinedCoulomb.has_value())
    {
        return std::unexpected(combinedCoulomb.error());
    }

    result.electrostatic = combinedCoulomb->totalEnergy - frozen->coulombEnergySum;

    const auto projectors = BuildOrthogonalizedChannels(frozen->scfs, overlapEigen, n);
    result.orthogonalizedDensityAlpha = projectors.alpha;
    result.orthogonalizedDensityBeta = projectors.beta;
    result.orthogonalizedDensity = projectors.alpha + projectors.beta;

    // E_Pauli of the run's own functional at the orthogonalized spin pair:
    // the energy the antisymmetrized, orthogonalized combination adds beyond
    // the frozen classical interaction (defined so the partition identity
    // holds). The seam is evaluated at the true spin pair - not at the
    // spin-summed density with the closed-shell split - which is what makes
    // the zero-XC/c_HF = 1 limit reproduce AnalyzeNocvEtsUnrestricted.
    auto orthEnergy = EvaluateKsSpin(
        molecule, coreHamiltonian, eri, projectors.alpha, projectors.beta, functional);

    if (!orthEnergy.has_value())
    {
        return std::unexpected(orthEnergy.error());
    }

    result.pauli = orthEnergy->totalEnergy - frozen->energySum - result.electrostatic;

    // The per-spin three-point (Simpson) transition state. With
    // D_sigma(lambda) = P_orth,sigma + lambda Delta D_sigma,
    //     E[D] - E[P_orth] = sum_sigma integral_0^1 dlambda
    //                        Tr[Delta D_sigma F_sigma(D(lambda))],
    // and Simpson's rule replaces each spin's integrand samples by
    //     F_sigma^TS = 1/6 F_sigma[P_orth] + 2/3 F_sigma[P_TS]
    //                  + 1/6 F_sigma[D].
    // The XC potential is evaluated at all three per-spin pairs, so a
    // non-linear functional enters every sample; for a quadratic functional
    // each weighted average collapses to the midpoint Fock exactly, which is
    // the Hartree-Fock limit the module's tests pin.
    const Eigen::MatrixXd transitionAlpha = 0.5 * (projectors.alpha + densityAlpha);
    const Eigen::MatrixXd transitionBeta = 0.5 * (projectors.beta + densityBeta);

    auto fockOrth = EvaluateKsSpin(
        molecule, coreHamiltonian, eri, projectors.alpha, projectors.beta, functional);

    if (!fockOrth.has_value())
    {
        return std::unexpected(fockOrth.error());
    }

    auto fockTransition =
        EvaluateKsSpin(molecule, coreHamiltonian, eri, transitionAlpha, transitionBeta, functional);

    if (!fockTransition.has_value())
    {
        return std::unexpected(fockTransition.error());
    }

    auto fockMolecular =
        EvaluateKsSpin(molecule, coreHamiltonian, eri, densityAlpha, densityBeta, functional);

    if (!fockMolecular.has_value())
    {
        return std::unexpected(fockMolecular.error());
    }

    const Eigen::MatrixXd transitionStateFockAlpha = (1.0 / 6.0) * fockOrth->fockAlpha +
                                                     (2.0 / 3.0) * fockTransition->fockAlpha +
                                                     (1.0 / 6.0) * fockMolecular->fockAlpha;
    const Eigen::MatrixXd transitionStateFockBeta = (1.0 / 6.0) * fockOrth->fockBeta +
                                                    (2.0 / 3.0) * fockTransition->fockBeta +
                                                    (1.0 / 6.0) * fockMolecular->fockBeta;

    // The two INDEPENDENT NOCV problems, one per spin, resolved against each
    // spin's own transition Fock in the Loewdin basis.
    const LowdinTransform lowdin = BuildLowdinTransform(overlapEigen);
    const Eigen::MatrixXd deltaAlpha = densityAlpha - projectors.alpha;
    const Eigen::MatrixXd deltaBeta = densityBeta - projectors.beta;
    const Eigen::MatrixXd fockAlphaLowdin =
        lowdin.rootInverse * transitionStateFockAlpha * lowdin.rootInverse;
    const Eigen::MatrixXd fockBetaLowdin =
        lowdin.rootInverse * transitionStateFockBeta * lowdin.rootInverse;
    const auto alphaChannels =
        ResolveChannels(lowdin.root * deltaAlpha * lowdin.root, fockAlphaLowdin);
    const auto betaChannels =
        ResolveChannels(lowdin.root * deltaBeta * lowdin.root, fockBetaLowdin);
    result.orbitalComponentsAlpha = alphaChannels.components;
    result.nocvEigenvaluesAlpha = alphaChannels.eigenvalues;
    result.nocvEigenvaluesAlphaNegative = alphaChannels.negativeEigenvalues;
    result.orbitalComponentsBeta = betaChannels.components;
    result.nocvEigenvaluesBeta = betaChannels.eigenvalues;
    result.nocvEigenvaluesBetaNegative = betaChannels.negativeEigenvalues;
    result.orbitalUnrestricted = alphaChannels.energy + betaChannels.energy;

    // The total deformation density is the sum of the two spin problems;
    // there is no restricted resolution on a per-spin reference to report in
    // the spin-summed fields, so orbitalComponents and nocvEigenvalues stay
    // empty (the AnalyzeNocvEtsUnrestricted convention).
    result.deformationDensityAlpha = deltaAlpha;
    result.deformationDensityBeta = deltaBeta;
    result.deformationDensity = deltaAlpha + deltaBeta;

    result.orbital = result.orbitalUnrestricted;
    result.bindingEnergy = result.electrostatic + result.pauli + result.orbital;
    return result;
}

} // namespace qcx::properties
