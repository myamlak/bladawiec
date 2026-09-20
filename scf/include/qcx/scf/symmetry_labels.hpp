#pragma once

/// \file
/// Full-group symmetry labels of a converged SCF state.
///
/// The a-posteriori labeling stage runs once after convergence: every MO is
/// labeled by its full-group irrep (chi-only projectors, real form),
/// degenerate partners are canonicalized by diagonalizing the Fock
/// restricted to their irrep subspace (skipped and recorded when the
/// subspace straddles the aufbau boundary), and the density is symmetrized
/// over the group's AO action. The SCF computation itself stays on the
/// Abelian reduction; nothing computed in the loop changes.
///
/// A label is REPORTED rather than guessed: a vector whose best
/// isotypic component does not carry essentially all the weight is labeled
/// kUnknownLabel and carries no irrep index, and LabelVector / LabelVectors
/// apply the SAME rule to a caller-supplied vector with no SCF state and no
/// Fock involved.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/symmetry/detection.hpp"
#include "qcx/symmetry/point_group.hpp"
#include "qcx/symmetry/point_group_name.hpp"

#include <Eigen/Dense>
#include <string>
#include <string_view>
#include <vector>

namespace qcx::scf {

/// The label a vector carries when its full-group irrep cannot be determined.
///
/// It is a LABEL VALUE and not an absent field on purpose: the io schema
/// serializes this vector as strings, so an omitted entry would read as a
/// shorter list and a guessed entry would read as a determination - and a
/// guess, a failed run and a silently omitted field are each worse than an
/// honest unknown.
inline constexpr std::string_view kUnknownLabel = "UNKNOWN";

/// The thresholds of the labelling rule, beside the rule so a caller can
/// read the numbers it was judged against instead of restating them.
///
/// The scoring quantity is a SQUARED norm: the score of irrep i on a vector v
/// is `v^T P_i v`, and every `P_i` is an orthogonal projector, so the score is
/// `||P_i v||^2` - the squared norm of v's isotypic component - and the scores
/// of a complete set sum to `||v||^2`. The rule is applied to the WEIGHT
/// FRACTION `bestScore / scoreSum`, which is the statement that one isotypic
/// component carries essentially all the weight.
///
/// Why 1e-6: the repository's matrix gates sit at 1e-8, but those bound norms
/// while this bounds a SQUARED-norm ratio, and the two failure modes it
/// separates are orders of magnitude apart (a determined vector scores ~1 to
/// machine precision; a near-tie scores ~0.5), so 1e-6 sits in a wide empty
/// band and the exact value is not critical.
inline constexpr double kLabelDeterminedTolerance = 1e-6;

/// The projector-consistency threshold: the sum of the isotypic scores must
/// reproduce the vector's own squared norm to this RELATIVE accuracy.
///
/// It is an identity for a complete orthogonal resolution of the identity
/// (`sum_i P_i = I`, which the character-orthogonality construction of the
/// projectors guarantees), so a failure is not an unknown: it means an
/// incomplete projector set or a wrong action, and it is reported as its own
/// outcome rather than folded into kUnknown.
inline constexpr double kLabelProjectorConsistencyTolerance = 1e-6;

/// The near-zero-norm floor: a vector whose squared norm is at or below this
/// has nothing to label and the weight fraction would be 0/0.
///
/// A squared norm, so it is a norm of 1e-6 - twelve orders below a normalized
/// MO's own norm of 1.
inline constexpr double kLabelMinVectorNormSquared = 1e-12;

/// What a labelling determined.
enum class LabelOutcome {
    /// One isotypic component carries essentially all the weight
    /// (kLabelDeterminedTolerance), so the label is that irrep's.
    kDetermined,
    /// No single irrep does: a NEAR-TIE between two irreps, or a vector spread
    /// over several. The label is kUnknownLabel - reporting the argmax here
    /// would name a winner decided by noise.
    kUnknown,
    /// The isotypic projectors do not resolve the identity on this vector
    /// (kLabelProjectorConsistencyTolerance). This is a DEFECT REPORT, not an
    /// unknown: the projector set is incomplete or the action is wrong.
    kProjectorDefect,
};

/// One labelled vector, with the measurements the outcome rests on.
///
/// Carried rather than recomputed because the numbers ARE the evidence for the
/// outcome: a reader who disagrees with a kUnknown can see which irrep nearly
/// won and by how much.
struct VectorLabel {
    /// The irrep label, or kUnknownLabel when the outcome is not kDetermined.
    std::string label = std::string(kUnknownLabel);
    /// Row into the full-group character table; -1 when the outcome is not
    /// kDetermined. (-1 also means "no finite table" on the linear-molecule
    /// path, which never reaches this type.)
    int irrepIndex = -1;
    /// What the labelling determined (see LabelOutcome).
    LabelOutcome outcome = LabelOutcome::kUnknown;

    /// The best isotypic score `max_i ||P_i v||^2`.
    double bestScore = 0.0;
    /// The score sum `sum_i ||P_i v||^2`, which equals `||v||^2` on a
    /// consistent projector set.
    double scoreSum = 0.0;
    /// The vector's own squared norm `||v||^2`.
    double normSquared = 0.0;
    /// `bestScore / scoreSum`, the weight fraction the outcome turns on; 0
    /// when scoreSum is 0.
    double weightFraction = 0.0;
    /// `|scoreSum - normSquared| / normSquared`, the projector-consistency
    /// residual; 0 when normSquared is at or below the floor.
    double projectorResidual = 0.0;
};

/// One canonicalization outcome for a degenerate-irrep subspace.
///
/// The canonicalization diagonalizes the Fock restricted to the subspace
/// spanned by the MOs of one irrep. The rotation is applied (and the
/// subspace listed in SymmetryLabels::canonicalized) only when the subspace
/// is entirely occupied or entirely virtual - the density is invariant
/// then. A subspace straddling the aufbau boundary (a half-filled HOMO, or
/// an irrep with both occupied and virtual MOs) is skipped and listed in
/// SymmetryLabels::straddled instead: the partners keep the SCF's arbitrary
/// rotation, and the record makes the arbitrariness explicit (the
/// honest-output rule).
struct DegenerateSubspaceRecord {
    std::string irrepLabel; ///< Full-group irrep label (e.g. "E", "T2g", "Pi_u").
    std::vector<int> moIndices; ///< MO columns of the subspace (coefficients order).
};

/// The full-group labeling output of one spin channel.
///
/// The RHF result carries one SymmetryLabels (the spin-summed density was
/// symmetrized); the UHF result carries one per spin. Labels are pure
/// classification - the SCF energy and the unsymmetrized state are
/// untouched; SymmetryLabels::density is the symmetrized density the
/// properties consumers use.
struct SymmetryLabels {
    qcx::symmetry::PointGroupName fullGroup = qcx::symmetry::PointGroupName::kC1;
    ///< The detected full group (the table path, or kCInfV/kDInfH on the
    ///< linear-molecule lambda path).
    qcx::symmetry::PointGroup abelianReduction = qcx::symmetry::PointGroup::kC1;
    ///< The computational (Abelian) group the SCF actually used.

    /// Per-MO Mulliken labels (one per coefficient column), e.g. "A1",
    /// "T2g" (table path) or "S+g", "Pi_u" (linear path, where g/u encode
    /// the inversion parity and the greek letter the |lambda| sector).
    std::vector<std::string> labels;
    /// Per-MO row index into the full-group character table; -1 on the
    /// linear path (no finite table exists there) and -1 where the outcome
    /// below is not kDetermined.
    std::vector<int> irrepIndices;
    /// Per-MO labelling outcome, one per coefficient column. Where an
    /// entry is not kDetermined the matching `labels` entry is kUnknownLabel
    /// and the matching `irrepIndices` entry is -1: the three agree, so a
    /// consumer reading any one of them cannot be told a guess.
    ///
    /// The linear-molecule path fills this with kDetermined: it reads the
    /// label off an operator's eigenvalue rather than scoring projectors, so
    /// there is no weight fraction to decide on.
    std::vector<LabelOutcome> labelOutcomes;

    /// The MO coefficients after degenerate-partner canonicalization: the
    /// columns of each rotated subspace were replaced by the Fock's
    /// eigenbasis within it (identity on converged states up to solver
    /// noise; the density is invariant by the aufbau-straddle rule).
    Eigen::MatrixXd coefficients;
    /// The symmetrized density D_sym = (1/|G|) sum_g A(g) D A(g)^T (over
    /// the full group, or over the finite symmetrization subset on the
    /// linear path). Energies are NOT recomputed from it.
    Eigen::MatrixXd density;

    /// Subspaces whose degenerate partners were canonicalized (rotated).
    std::vector<DegenerateSubspaceRecord> canonicalized;
    /// Subspaces straddling the aufbau boundary: recorded, NOT rotated.
    std::vector<DegenerateSubspaceRecord> straddled;

    /// The elements the density was averaged over (linear path only): the
    /// closure of the recorded finite operations of the detected group -
    /// the continuum symmetries (C-inf-v / D-inf-h) cannot be finitely
    /// averaged, and the subset used is documented here. Empty on the table
    /// path (the average runs over the full group).
    std::vector<std::string> symmetrizationSubset;
    /// |G| on the table path, or the finite subset size on the linear path.
    int averagedElementCount = 0;
};

/// Labels ONE caller-supplied vector in the FULL group, off the SCF path.
///
/// The labelling stage's own contract is the converged MO coefficient matrix of
/// an SCF state, energy-sorted, produced by the loop. This entry point has no
/// SCF state and no Fock, and labels a vector the CALLER built - a natural
/// orbital, a CI vector, a response vector, an excited-state root. The rule is
/// the one the SCF path applies (LabelOutcome, VectorLabel), so a root and an
/// MO column are judged by the same threshold rather than by two.
///
/// The result is a REPORT: nothing is projected, nothing is rotated, no state
/// is read and no energy is computed. The group is realized from the molecule
/// and the analysis exactly as the labelling stage does, so the two agree by
/// construction rather than by a promise.
///
/// \param vector The vector to label, in the AO basis `basisSet` spans, with
///   that basis's function count rows. It is used as given: the label is the
///   one THIS vector carries, which is the caller's to interpret.
/// \param molecule The molecule the labelling is about.
/// \param basisSet The basis the vector's rows index.
/// \param analysis The symmetry analysis of `molecule` (the full group is
///   realized from it).
/// \returns The label and the measurements behind it, or an Error:
///  - kUnimplemented: the detected full group is one of the LINEAR groups
///    (C-inf-v / D-inf-h), where no finite character table exists and the
///    label is read off the |lambda| operator instead of scored from isotypic
///    projectors - so the rule this entry point documents is not the rule in
///    force there, and it is refused BY NAME rather than answered under a rule
///    that does not apply.
///  - kInvalidArgument: the vector's length does not match the basis, the
///    analysis is inconsistent with the molecule, or the realized action is not
///    a representation of the closure.
qcx::Result<VectorLabel> LabelVector(const Eigen::VectorXd& vector,
                                     const qcx::molecule::Molecule& molecule,
                                     const qcx::basisset::BasisSet& basisSet,
                                     const qcx::symmetry::SymmetryAnalysis& analysis);

/// Labels SEVERAL caller-supplied vectors in the FULL group, off the SCF path,
/// building the group realization and the isotypic projectors ONCE.
///
/// The batch form exists for the caller who holds many vectors - a response
/// manifold, a set of natural orbitals - because one realization and one
/// projector set cost O(|G| n^3) and rebuilding them per vector is the same
/// waste the blocked diagonalization's once-per-run data exists to avoid
/// (BlockedDiagonalizeData). Each vector then costs O(|G| n^2).
///
/// \param vectors One vector per COLUMN, in the AO basis.
/// \returns One VectorLabel per column, in column order, or an Error with its
///   reasons named under LabelVector.
qcx::Result<std::vector<VectorLabel>> LabelVectors(const Eigen::MatrixXd& vectors,
                                                   const qcx::molecule::Molecule& molecule,
                                                   const qcx::basisset::BasisSet& basisSet,
                                                   const qcx::symmetry::SymmetryAnalysis& analysis);

} // namespace qcx::scf
