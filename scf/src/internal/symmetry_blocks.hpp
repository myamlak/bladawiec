#pragma once

// AO-space symmetry decomposition and blocked diagonalization, first pass
// (diagonalization-only scope): when the computational point
// group is a non-trivial Abelian group, the AO-space Fock matrix is
// block-diagonal in the symmetrized basis, and diagonalizing block-wise is
// mathematically equivalent to the plain n x n solve. Internal to qcx-scf;
// wired into the RHF/UHF loops through the FockBuilderFn overloads'
// optional BasisSet argument, which the driver supplies. The
// SalcSet/GenerateSalcs machinery of the symmetry module is atom-space and
// deliberately not reused here.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/scf/symmetry_labels.hpp"
#include "qcx/symmetry/detection.hpp"
#include "qcx/symmetry/full_group_tables.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

namespace qcx::scf::internal {

// The AO-space symmetry decomposition of one molecule.
//
// u holds the symmetrized basis as orthonormal columns, ordered irrep-major:
// the first blockSizes[0] columns span the first irrep of the computational
// group, the next blockSizes[1] the second, and so on. In that basis every
// AO-space matrix that commutes with the group's exact action (the Fock,
// overlap, H_core, ...) becomes block-diagonal up to floating-point noise:
// ||(U^T M U)_offdiag|| is OffBlockNorm(M, blocks).
struct SymmetryBlocks {
    Eigen::MatrixXd u; // n x n orthonormal, irrep-major column order.
    std::vector<Eigen::Index> blockSizes; // One entry per irrep block.
    std::vector<std::vector<std::size_t>> aoOrbits; // Minimal invariant subsets of AO indices.
    bool isTrivial = false; // C1 (or unsupported): u = I and one block of size n.

    // The generators' action on the symmetrized basis, one entry per
    // generator of the computational group (BuildGroupData's order): the sign
    // s_i = +-1 with A(g_j) u_i = s_i u_i for every column i of u. The groups
    // here are Abelian with one-dimensional irreps, so this action is a
    // scalar on every irrep block and U^T A(g_j) U is EXACTLY
    // diag(generatorSigns[j]) - which is what lets the adaptation guard below
    // measure the commutator in O(n^2) per generator instead of an n^3
    // product. Extracted from the realized action rather than from the
    // character table, so the guard's reference is the same realization the
    // projectors were built from; every entry is validated against that
    // realization at build time. Empty for the trivial decomposition (C1 has
    // no generators).
    std::vector<Eigen::VectorXd> generatorSigns;
};

// Builds the AO-space decomposition of a molecule's basis for its detected
// computational group.
//
// The generators of the computational group are realized from the verified
// elements of the analysis (rotations by pi about the distinct even-order
// rotation axes, the distinct mirrors, and inversion when present); each
// generator induces an atom permutation (greedy distance matching about the
// center of mass, like symmetry's own matcher) and an exact angular action
// on every shell. The group is the set of subset products of the generators
// (every generator is an involution and the group is Abelian); its irreps
// are the distinct sign patterns on the generators. The projector
// P = (1/|G|) sum_g chi(g) A(g) is an orthogonal projector onto the irrep
// subspace, and its orthonormal range is one block of u.
//
// \returns The decomposition, or an Error:
//  - kUnimplemented: the computational group is not realizable from the
//    detected elements (or not one of the eight Abelian groups) - callers
//    fall back to the plain path.
//  - kInvalidArgument: the analysis is inconsistent with the molecule (an
//    operation does not permute the atoms, or a basis entry is missing).
qcx::Result<SymmetryBlocks> BuildSymmetryBlocks(const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& basisSet,
                                                const qcx::symmetry::SymmetryAnalysis& analysis);

// The AO-space symmetry reduction for the petite list: the
// action of the computational group on the basis functions as a signed
// permutation, extracted from the same realized elements BuildSymmetryBlocks
// uses. The petite list requires every element's AO action to be a signed
// permutation in the global function basis - true exactly when the realized
// element matrices are signed coordinate permutations of the global frame
// (the molecule's symmetry planes/axes are the global coordinate planes/
// axes). Elements that meet the gate are kept, and when some element does
// not the reduction falls back to the largest axis-aligned subgroup of the
// computational group (its elements still commute with F, so the class
// contraction stays exact); groupOrder then reports the subgroup. When only
// the identity aligns the reduction is inapplicable.
//
// \returns The reduction, or an Error:
//  - kUnimplemented: the computational group is not realizable (delegated
//    from BuildGroupData) or no non-identity element aligns - the caller
//    falls back to the plain path.
//  - kInvalidArgument: the analysis is inconsistent with the molecule.
qcx::Result<qcx::integrals::SymmetryReduction> BuildSymmetryReduction(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::symmetry::SymmetryAnalysis& analysis);

// Blocked sibling of DiagonalizeFock: diagonalizes U^T (X^T F X) U block by
// block with SelfAdjointEigenSolver, embeds the per-block eigenvectors into
// the full space, globally re-sorts by ascending eigenvalue (the plain path
// sorts globally; per-block sorts would not match it), and returns
// C = X U W. For kC1 and unsupported groups (kUnimplemented from
// BuildSymmetryBlocks) it delegates to the plain DiagonalizeFock; other
// errors propagate.
qcx::Result<Eigen::MatrixXd> DiagonalizeFockBlocked(
    const Eigen::MatrixXd& fock,
    const Eigen::MatrixXd& x,
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::symmetry::SymmetryAnalysis& analysis);

// The once-per-run per-irrep transform data of the blocked diagonalization.
// X = S^{-1/2} commutes with the group action (S is
// group-invariant), so U^T X U is block-diagonal up to integral noise and
// the per-iteration transform can run per irrep: g_b = X_bb^T (U_b^T F U_b)
// X_bb, back-transform C = (U X_diag W) P. Built once per SCF run next to
// BuildSymmetryBlocks - rebuilding per iteration would redo n^3 work.
struct BlockedDiagonalizeData {
    /// Mirror of SymmetryBlocks::isTrivial: for trivial groups the
    /// diagonalization delegates to the plain path and ignores the blocks.
    bool isTrivial = false;
    /// Per nonzero irrep block (size-0 blocks skipped), in block order: the
    /// column range [offsets[i], offsets[i] + sizes[i]) of SymmetryBlocks::u.
    std::vector<Eigen::Index> offsets;
    /// Per block: its size (the range above; the zero blocks of
    /// SymmetryBlocks::blockSizes do not appear).
    std::vector<Eigen::Index> sizes;
    /// Per block: X_bb = U_b^T X U_b, the diagonal block of U^T X U.
    std::vector<Eigen::MatrixXd> xBlocks;
    /// Y = X U (n x n), the transform from the AO basis straight to the
    /// symmetrized ORTHONORMAL basis, built once per run: M = Y^T F Y is the
    /// Fock in that basis, U^T (X^T F X) U, at two multiplies instead of
    /// four. The adaptation guard is its only consumer. Empty for trivial
    /// blocks (the guard is not reached there - the diagonalization delegates
    /// to the plain path).
    Eigen::MatrixXd symmetrizedBasis;
};

/// Builds the per-run transform data once (the SCF loop builds
/// SymmetryBlocks ONCE per run, and the data rides along). Trivial
/// blocks yield trivial data with empty vectors. The caller's x and blocks
/// come from the same molecule/basis (both call sites build them from one
/// run), so they are always square and consistent.
/// \param x The orthogonalizer X = S^{-1/2}.
/// \param blocks The once-per-run blocks.
/// \returns The data.
qcx::Result<BlockedDiagonalizeData> BuildBlockedDiagonalizeData(const Eigen::MatrixXd& x,
                                                                const SymmetryBlocks& blocks);

// The per-iteration form: diagonalizes against already-built blocks (the
// SCF loop builds SymmetryBlocks and the BlockedDiagonalizeData ONCE per run
// and reuses them every iteration; rebuilding per iteration
// would redo the group realization, the projector eigensolves, and the
// U^T X U blocks, dominating any diagonalization saving). The per-irrep
// transforms and solves run on the OpenMP team (the threaded
// eigensolve exception to the BLAS-sequential rule - the region holds
// Eigen kernels only, never nests BLAS, and the results are bit-identical
// across team sizes). Trivial blocks (kC1) delegate to the plain
// DiagonalizeFock.
//
// THE EQUIVALENCE IS CONDITIONAL, and this function assumes the condition
// as a precondition: blocking is equivalent to the plain solve exactly while
// the Fock commutes with the group. Its caller owns that test - measure the
// Fock with MeasureSymmetryAdaptation and fall back to DiagonalizeFock when
// it fails (the scf loop's per-spin rule). Handing it a non-commuting Fock
// does not fail loudly: it solves a different problem, the one constrained
// to the group's irreps, and returns that answer.
qcx::Result<Eigen::MatrixXd> DiagonalizeFockBlocked(const Eigen::MatrixXd& fock,
                                                    const Eigen::MatrixXd& x,
                                                    const SymmetryBlocks& blocks,
                                                    const BlockedDiagonalizeData& data);

// How far one Fock matrix is from being symmetry-adapted - the blocked
// solve's precondition, measured rather than assumed.
//
// The equivalence blocking rests on ("U^T (X^T F X) U is block-diagonal up to
// integral noise ... mathematically equivalent to the plain n x n solve")
// holds exactly while F commutes with the group's action. That is a property
// of the SOLUTION, not of the molecule: the point group is a property of the
// NUCLEAR FRAMEWORK, while the group that may legitimately constrain a
// solution is the invariance group OF THAT SOLUTION, and an unrestricted
// solution is free to have a smaller one. When it does, blocking by the full
// point group does not accelerate that solution's diagonalization - it
// PROJECTS THE SOLUTION OUT.
//
// What this measures is "this Fock is not symmetry-adapted", which is what
// makes blocking invalid. It is a NECESSARY condition and NOT a detector of
// broken symmetry: a small commutator does not show that no lower
// symmetry-broken solution exists - it shows that the matrix about to be
// blocked is (numerically) adapted, which can hold while a broken solution
// sits elsewhere on the surface. Read it as a precondition, never as a
// variational statement.
struct SymmetryAdaptation {
    /// || [F_orth, A(g_j)] ||_F / || F_orth ||_F for every generator g_j of
    /// SymmetryBlocks::generatorSigns, in the same order. Relative to the
    /// Fock's own Frobenius norm so one tolerance serves every molecule and
    /// basis (an absolute norm scales with the Hamiltonian's size and would
    /// need a per-system threshold). ~1e-12..1e-14 for an adapted Fock; ~1
    /// for a solution that has broken the generator.
    std::vector<double> generatorCommutatorNorms;
    /// The largest entry of generatorCommutatorNorms; 0 when there is none
    /// (no generators, or a zero Fock - nothing can be violated).
    double maxCommutatorNorm = 0.0;
};

/// The threshold the relative commutator norm is compared against, and the
/// home of the number (callers record the value they used beside their
/// result rather than restating it).
///
/// It is the repo's EXISTING published noise level for exactly this property:
/// OffBlockNorm above documents ~1e-12..1e-10 for a matrix commuting with the
/// group and the symmetry tests gate at 1e-8. The two failure modes are
/// asymmetric - too loose BLOCKS a non-adapted Fock and silently constrains
/// the variational search, too tight only LOSES THE SPEEDUP - so the guard
/// errs toward disengaging and the threshold sits at the low end of the
/// defensible band rather than the high end. The measured separation on the
/// canonical broken-symmetry case is many orders wide on both sides (H2/STO-3G
/// at R = 3.5 bohr: the converged Fock's norm is ~1; the adapted control at
/// R = 1.4 bohr is ~1e-13), which is what makes the exact value uncritical.
inline constexpr double kSymmetryAdaptationTolerance = 1e-8;

// The energy tolerance that separates two DEGENERATE MANIFOLDS of one irrep.
//
// The canonicalization groups degenerate partners by clustering them on ENERGY
// within this tolerance FIRST and by irrep label only after that. The label
// alone is not a key: two spatially separate manifolds can share one irrep
// label (ammonia's E irrep of C3v carries two partner pairs), and grouping them
// under the label hands the aufbau-straddle test one four-column subspace where
// the truth is two two-column ones that each sit entirely on one side of the
// boundary - so a pair that should have been canonicalized is reported as
// straddled and skipped (measured on
// `AmmoniaC3vCanonicalization`).
//
// The number: converged degenerate partners agree to integral noise or better
// and the tests pin them 1e-8 apart, while distinct manifolds of one irrep are
// separated by ~1e-1 Ha; 1e-6 Ha is two decades above the former and five below
// the latter. It is an ABSOLUTE comparison in Hartree, so it needs no
// per-system reference to be read.
inline constexpr double kManifoldEnergyTolerance = 1e-6;

/// Measures the blocked-solve precondition of one Fock (see SymmetryAdaptation).
///
/// The commutator is taken in the ORTHONORMAL basis, F_orth = X^T F X, and
/// evaluated in the symmetrized basis, where every generator's action is
/// exactly the signed diagonal diag(SymmetryBlocks::generatorSigns[j]). The
/// identity
///   || [F_orth, A(g)] ||_F = || [U^T F_orth U, U^T A(g) U] ||_F
/// holds because U is orthogonal, and the right-hand side costs one n x n
/// transform for the Fock (M = Y^T F Y, Y = X U, precomputed) plus O(n^2) per
/// generator instead of two n^3 products each.
///
/// \param fock The Fock matrix in the AO basis (the matrix about to be
///   diagonalized).
/// \param blocks The once-per-run decomposition; its generatorSigns are the
///   reference. A trivial decomposition yields an empty measurement.
/// \param data The once-per-run transform data (its symmetrizedBasis is the
///   Y above).
/// \returns The per-generator norms and their maximum.
SymmetryAdaptation MeasureSymmetryAdaptation(const Eigen::MatrixXd& fock,
                                             const SymmetryBlocks& blocks,
                                             const BlockedDiagonalizeData& data);

// The off-block invariant: the relative infinity norm of the off-block part of
// U^T matrix U (the diagonal blocks are zeroed, the remainder is divided by
// the matrix's max |entry|). ~1e-12..1e-10 for a matrix commuting with the
// group; the tests assert < 1e-8.
double OffBlockNorm(const Eigen::MatrixXd& matrix, const SymmetryBlocks& blocks);

// ---- Full-group labeling stage (a posteriori) ----

// One realized element of the full-group closure: the 3x3
// matrix and its runtime classification, mirroring the table generator's
// classify() (symmetry_blocks.cpp; the axis is canonicalized, first
// nonzero component positive).
struct FullGroupElement {
    Eigen::Matrix3d matrix;
    qcx::symmetry::OperationKind kind = qcx::symmetry::OperationKind::kIdentity;
    int order = 1; // Minimal power g^k = E; 1 for mirrors (the table convention).
    int power = 1; // j in C_n^j / S_n^j (class keys); 1 for identity/inversion/mirrors.
    Eigen::Vector3d axis = Eigen::Vector3d::Zero(); // Canonicalized; zero for identity/inversion.
};

// The runtime classes of the closure: the conjugacy classes with
// their (kind, order, j, slot) table keys, in the generator's assignment
// order. The slot is the axis role: kPrincipal for the maximal-order proper
// rotation (ties: the class-minimal power, then the D2-family rule - the
// axis carrying an improper order-4 class wins, else the axis whose
// elements fix the most atoms, most-fixing mirror first - then the
// lexicographically smallest canonical axis) and for sigma/improper classes
// on its axis; kOrbitN ranked by the lexicographically smallest
// frame-canonicalized member axis among same-(kind, order, j) classes;
// kNone for identity/inversion.
struct FullGroupClass {
    qcx::symmetry::OperationKind kind = qcx::symmetry::OperationKind::kIdentity;
    int order = 1;
    int power = 1;
    qcx::symmetry::ClassAxis slot = qcx::symmetry::ClassAxis::kNone;
    Eigen::Vector3d minAxis = Eigen::Vector3d::Zero(); // Lex-min canonical member axis.
};

// The realization as returned by BuildFullGroupRealization (declared here so
// the return type is complete at the call site's declaration; the definition
// follows, top-down).
struct FullGroupRealization;

// The realized full-group closure of a molecule: the elements
// of the analysis plus the powers of the recorded rotations and improper
// rotations, closed under multiplication, deduplicated at 1e-8 (identity
// first), classified, partitioned into conjugacy classes, and realized as
// AO-space matrices A(g). The representation guard runs here: A(g) A(h) =
// A(gh) on the closure-defining (g, generator) pairs, which inductively
// certifies the whole closure. The closure order is NOT checked against the
// character table here - the finite path asserts it in the stage (the
// linear groups have no table).
//
// \returns The realization, or an Error:
//  - kInvalidArgument: the closure is inconsistent with the analysis (an
//    operation does not permute the atoms, or the realized action is not a
//    representation of the closure).
qcx::Result<qcx::scf::internal::FullGroupRealization> BuildFullGroupRealization(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::symmetry::SymmetryAnalysis& analysis);

// The realization as returned above (declared after BuildFullGroupRealization
// so the struct order reads top-down).
struct FullGroupRealization {
    std::vector<FullGroupElement> elements; // Closure order, identity first.
    std::vector<Eigen::MatrixXd> actions; // A(g) for each element.
    std::vector<FullGroupClass> classes; // The runtime classes.
    std::vector<int> classOfElement; // Class index of every element.
    int groupOrder = 0; // The closure order.
};

// The exact z-axis angular-momentum-squared operator (the square of the z
// component of L) on one shell block, in the engine's function order (used
// by the linear-molecule labeling path): the diagonal (index - l)^2 on spherical
// functions, the closed-form monomial map
//   Lz^2 x^a y^b z^c = (2ab + a + b) x^a y^b z^c
//                      - a(a-1) x^{a-2} y^{b+2} z^c - b(b-1) x^{a+2} y^{b-2} z^c
// on Cartesian ones. Its eigenvalues are the squares of the magnetic quantum
// numbers m = index - l (spherical) or of the m values of the monomials
// (Cartesian), which the lambda path reads off column-wise.
Eigen::MatrixXd AngularMomentumSquaredAboutZ(int l, bool isSpherical);

// The full-group labeling stage: labels every MO of a converged state by
// its full-group irrep (the
// chi-only real-form projectors, argmax score), canonicalizes the degenerate
// partners by diagonalizing the Fock restricted to each dim >= 2 irrep
// subspace - applied only when the subspace is entirely occupied or entirely
// virtual (the density is invariant then), skipped and recorded when it
// straddles the aufbau boundary - and symmetrizes the density over the
// group's AO action. The SCF computation itself stays on the Abelian
// reduction; the inputs are left untouched and the energy is not recomputed.
//
// The linear-molecule groups (C-inf-v / D-inf-h) have no finite character
// table: the label is built from |lambda| (the z-component-squared operator
// about the principal axis) plus the inversion parity (g/u) and the sigma-v
// parity (Sigma+/-), and the density is averaged over the recorded finite
// subset (the continuum symmetries cannot be finitely averaged; the subset
// is documented in SymmetryLabels::symmetrizationSubset).
//
// \param fock The converged Fock matrix (canonicalization only; the density
//   symmetrization does not use it).
// \param coefficients The converged MO coefficients, energy-sorted columns.
// \param density The density of the state (spin-summed for RHF).
// \param occupiedCount The number of occupied MOs (the columns
//   0..occupiedCount-1 of coefficients).
// \returns The labels and the canonicalized/symmetrized state, or an Error:
//  - kInvalidArgument: the closure does not reproduce the table's order, the
//    realized action is not a representation, or an operation does not
//    permute the atoms.
qcx::Result<qcx::scf::SymmetryLabels> SymmetryLabelAndSymmetrize(
    const Eigen::MatrixXd& fock,
    const Eigen::MatrixXd& coefficients,
    const Eigen::MatrixXd& density,
    int occupiedCount,
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::symmetry::SymmetryAnalysis& analysis);

// ---- The shared labelling core ----

// The full-group isotypic projectors of one realization, in table irrep order,
// with the table's own labels.
//
// They are the chi-only real-form projectors `P_i = (dim / (|G| nu)) sum_g
// chi_i(class g) A(g)`, idempotent and self-adjoint, and their defining
// identity is that they RESOLVE THE IDENTITY: `sum_i P_i = I` by
// character-column orthogonality, so `sum_i ||P_i v||^2 = ||v||^2` for every
// vector. The labelling's consistency gate measures exactly that identity, and
// its failure is a defect rather than an unknown.
//
// Built once per consumer: one projector set costs O(|G| n^3), which is why the
// arbitrary-vector entry point labels a whole matrix against one set rather
// than rebuilding per vector.
struct IsotypicProjections {
    std::vector<Eigen::MatrixXd> projectors; // One per table irrep, table order.
    std::vector<std::string> labels; // The table's irrep labels, same order.
};

// Builds the projectors of a realized full-group closure against its character
// table.
//
// The runtime classes are matched to the table classes by their (kind, order,
// power, slot) keys with the class sizes verified, exactly as the labelling
// stage did inline before this function existed - the extraction is a move, not
// a rewrite, so the labelling stage and the arbitrary-vector entry point share
// one construction rather than two that could drift.
//
// \returns The projectors, or an Error:
//  - kInvalidArgument: a runtime class matches no table class, or a runtime
//    class size differs from the table's.
qcx::Result<IsotypicProjections> BuildIsotypicProjections(
    const FullGroupRealization& realization, const qcx::symmetry::FullGroupTable& table);

// The labelling rule, applied to one vector against one projector set.
//
// The score of irrep i is `v^T P_i v`, which is `||P_i v||^2` because P_i is an
// orthogonal projector; the outcome follows the thresholds documented beside
// them in the public header (LabelOutcome, VectorLabel, and the three
// kLabel* constants):
//  - PROJECTOR DEFECT when the scores do not reproduce `||v||^2` to
//    kLabelProjectorConsistencyTolerance relative accuracy;
//  - UNKNOWN when `||v||^2` is at or below kLabelMinVectorNormSquared (nothing
//    to label), or when the best score does not carry at least
//    `1 - kLabelDeterminedTolerance` of the score sum - which is the NEAR-TIE
//    case as well as the spread one;
//  - DETERMINED otherwise, with the label of the best-scoring irrep.
//
// \param vector The vector, in the AO basis the projectors act on.
// \param projectors The isotypic projectors (IsotypicProjections::projectors).
// \param labels Their labels, one per projector, in the same order.
qcx::scf::VectorLabel ClassifyVectorLabel(const Eigen::VectorXd& vector,
                                          const std::vector<Eigen::MatrixXd>& projectors,
                                          const std::vector<std::string>& labels);

} // namespace qcx::scf::internal
