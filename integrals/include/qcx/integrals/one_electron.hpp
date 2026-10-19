#pragma once

#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <array>
#include <span>
#include <vector>

namespace qcx::integrals {

/// \file
/// One-electron integral matrices, general angular momentum (l <= 6, the
/// parser cap).
///
/// The matrices are built through the matrix-form McMurchie-Davidson
/// machinery ([McMurchie1978]) at runtime angular momentum (no per-class
/// instantiations): overlap and kinetic as
/// small contractions of the Hermite E coefficients, nuclear attraction
/// with early charge summation (the Boys seeds are summed over nuclei first
/// through the 1-center VRR, then one bra contraction runs;
/// internal/md_one_electron.hpp).
///
/// The derivatives of the three matrices with respect to the nuclear
/// coordinates share that machinery: ComputeOneElectronGradient walks the same
/// pair data and reads each pair's differentiated blocks from the derivative
/// tier (internal/md_derivative.hpp), contracting them with a fixed density.
///
/// Every matrix is produced directly into a rank-2 CPU Tensor with
/// shape {n, n}, where n is the number of functions ordered by canonical
/// atom order, then shell order, then contraction row, then angular
/// component (shell_pairs.hpp). The tensors leave here host-canonical and
/// ready for \c WithDevice migration downstream.
///
/// Behavior change from the s-only engines: the kUnimplemented
/// rejection of non-s shells is replaced by the l > 6 parser-cap
/// rejection - p/d/... shells are evaluated positively.

/// Builds the overlap matrix S_uv = <u|v> over all functions.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell must satisfy l <= 6.
/// \returns The overlap matrix, or an Error (kUnimplemented for a shell
/// beyond the parser cap, kInvalidArgument when an atom has no basis
/// entry).
/// \ingroup qcx-integrals
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildOverlapMatrix(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet);

/// Builds the kinetic-energy matrix T_uv = -1/2 <u|laplace|v> over all
/// functions.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell must satisfy l <= 6.
/// \returns The kinetic matrix, or an Error (kUnimplemented for a shell
/// beyond the parser cap, kInvalidArgument when an atom has no basis
/// entry).
/// \ingroup qcx-integrals
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildKineticMatrix(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet);

/// Builds the nuclear-attraction matrix V_uv = <u| -sum_c Z_c / |r - R_c| |v>.
/// \param molecule Molecule providing the atom coordinates (Bohr) and charges Z.
/// \param basisSet Basis set; every shell must satisfy l <= 6.
/// \returns The nuclear-attraction matrix, or an Error (kUnimplemented for a
/// shell beyond the parser cap, kInvalidArgument when an atom has no basis
/// entry).
/// \ingroup qcx-integrals
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildNuclearAttractionMatrix(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet);

/// The SETUP's peak resident bytes: the never-under bound of everything a
/// run path pays before its first gate - the ramp this header's matrix
/// builds walk, PLUS the Schwarz screening sweeps every budgeted builder's
/// Create runs in front of its own budget decision.
///
/// The setup is what EVERY run path pays before its first gate, so a cap
/// below its footprint kills the process at the first over-cap touch inside
/// it, pre-iteration, with no admission decision of any kind reached (the
/// above-ceiling crash class; measured at C50H102/def2-SVP, 1210 basis
/// functions). The lean member's own Create-time envelope already charges
/// this setup as its runStoreBytes term (EstimatePeakBytes,
/// lean_fock_build.cpp); this function is that term for the families whose
/// builder has no envelope of its own to consult.
///
/// The terms - the geometry-only skeleton, one pair-data chunk, the
/// caller's live n x n matrices, and one SchwarzSweepBytes per basis in
/// effect - are the composition of internal::SetupPeakBytes (footprint.hpp),
/// which is the one home of the arithmetic; the bound is never under the
/// setup's allocation by construction (see its note). The sweep term is
/// what the earlier ramp-only bound omitted, and its absence was MEASURED:
/// at 1.0 GiB the RI-J link's absent-key leg died 0xC0000409 with no
/// diagnostic while this bound's ramp-only form reported only 483833976 B
/// against a 1690.7 MiB setup peak.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set (the primitive-count authority).
/// \param liveMatrices The n x n matrices the caller holds live across the
/// ramp (2 for the driver's T + V, then H beside S, sequence).
/// \param auxBasisSet The auxiliary basis set the run's wiring put in
/// effect, or null when the family consumes no aux (qcx::io::BuilderConsumesAux)
/// - null charges no aux sweep, because none runs.
/// \returns The peak resident bytes, or an Error from the canonical
/// shell-pair build (the same error the ramp's own builders report).
/// \ingroup qcx-integrals
qcx::Result<double> SetupRampPeakBytes(const qcx::molecule::Molecule& molecule,
                                       const qcx::basisset::BasisSet& basisSet,
                                       const qcx::basisset::BasisSet* auxBasisSet,
                                       std::size_t liveMatrices);

/// The setup admission's slack: the last-resort multiplier the engine already
/// applies to its own never-under envelopes (lean_fock_build.cpp kRefusalSlack,
/// the lean member's ceiling). It is stated here, once, because BOTH floors
/// below and above it read it - the driver's pre-gate refusal and the workspace
/// grant it reserves - and a second spelling is the drift this pair exists to
/// prevent.
/// \ingroup qcx-integrals
inline constexpr double kSetupAdmissionSlack = 1.5;

/// THE ONE SOURCE of the reserve a workspace budget must leave the run: the
/// bytes a run has committed OUTSIDE any builder's workspace budget by the time
/// that builder's budget decision is taken, being the setup ramp and the
/// Schwarz sweeps every budgeted builder runs in front of its own decision.
///
/// Both consumers read THIS: the driver's pre-gate admission refuses when the
/// reserve does not fit the memory cap, and the workspace it then builds is
/// granted the cap MINUS the same number. The two were one quantity in the
/// deleted predictive model (its base term was both floors at once) and were
/// split when that model went: the admission half was re-derived from this
/// header, the reservation half was not, and every budgeted rung was handed the
/// whole cap - leaving the cheaper rungs (RiFullFockRung::kBlocked, the RI-J
/// disk rung) unreachable through the driver at every cap.
///
/// It is \ref SetupRampPeakBytes times \ref kSetupAdmissionSlack, so it is never
/// under the setup's own allocation and it carries no fitted constant: an
/// earlier draft restored the deleted model's measured 0.056 GiB baseline as a
/// floor here, and that was refused - a number fitted to a non-final state has
/// no worth, and a copy of it is the same number in new clothes.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set (the primitive-count authority).
/// \param liveMatrices The n x n matrices the caller holds live across the ramp.
/// \param auxBasisSet The auxiliary basis set in effect, or null (nothing to
/// charge: no aux sweep runs).
/// \returns The reserve in bytes, or the shell-pair build's own error.
/// \ingroup qcx-integrals
qcx::Result<double> SetupAdmissionReserveBytes(const qcx::molecule::Molecule& molecule,
                                               const qcx::basisset::BasisSet& basisSet,
                                               const qcx::basisset::BasisSet* auxBasisSet,
                                               std::size_t liveMatrices);

/// Builds the nuclear-attraction matrix with only the masked centers'
/// charges: V_uv = <u| -sum_{c in centerIndices} Z_c / |r - R_c| |v>, in
/// the FULL basis of the molecule's atoms. The ETS-NOCV fragment embedding
/// needs the fragment's own potential over the molecular function space
/// (the shell list and the one-electron matrices of a fragment molecule
/// span only its own functions).
/// \param molecule Molecule providing the atom coordinates (Bohr) and
/// charges; ALL of its atoms supply the shell list (the full basis).
/// \param basisSet Basis set; every shell must satisfy l <= 6.
/// \param centerIndices The atom indices whose charges enter the
/// attraction sum; must be non-empty (a zero-matrix result would silently
/// hide a caller bug, so the empty list is rejected).
/// \returns The nuclear-attraction matrix, or an Error (kInvalidArgument
/// for an empty center list, an out-of-range index, or an atom without a
/// basis entry, kUnimplemented for a shell beyond the parser cap).
/// \ingroup qcx-integrals
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildNuclearAttractionMatrix(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    std::span<const std::size_t> centerIndices);

/// Builds the three Cartesian dipole-integral matrices D_k(u, v) =
/// <u| r_k - R_k |v>, one per direction (index 0 = x, 1 = y, 2 = z), each
/// {n, n} symmetric, through the Hermite moment machinery
/// ([Helgaker2000] Sec 9.5.1; internal/md_one_electron.hpp).
/// \param origin The reference point R (Bohr) the operator is measured
/// from; defaults to the coordinate origin. Origin convention: all moments
/// are relative to the input coordinate origin AS GIVEN - the caller is
/// responsible for recentering the molecule first if a center-of-mass or
/// center-of-charge dipole is wanted.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell must satisfy l <= 6.
/// \returns The three matrices, or an Error (kUnimplemented for a shell
/// beyond the parser cap, kInvalidArgument when an atom has no basis
/// entry).
/// \ingroup qcx-integrals
qcx::Result<std::array<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>, 3>> BuildDipoleMatrix(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const std::array<double, 3>& origin = {0.0, 0.0, 0.0});

/// Builds the six upper-triangle quadrupole-integral matrices
/// Q_kl(u, v) = <u| (r_k - R_k)(r_l - R_l) |v> for kl = xx, xy, xz, yy,
/// yz, zz (component order), each {n, n} symmetric. The off-diagonal kl
/// and lk integrals coincide, so only the upper triangle is built (the
/// quadrupole-symmetrization fix).
/// \param origin The reference point R (Bohr) the operator is measured
/// from; defaults to the coordinate origin. Same origin convention as
/// BuildDipoleMatrix.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell must satisfy l <= 6.
/// \returns The six matrices, or an Error (kUnimplemented for a shell
/// beyond the parser cap, kInvalidArgument when an atom has no basis
/// entry).
/// \ingroup qcx-integrals
qcx::Result<std::array<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>, 6>>
BuildQuadrupoleMatrix(const qcx::molecule::Molecule& molecule,
                      const qcx::basisset::BasisSet& basisSet,
                      const std::array<double, 3>& origin = {0.0, 0.0, 0.0});

/// Builds the electron potential of the density at arbitrary points,
///     V_elec(p) = -sum_uv D_uv <u| 1/|r - p| |v>,
/// the ESP-fit observable (CHELPG/MK; the nuclear part sum_A Z_A / |p - R_A|
/// is analytic and stays on the caller's side). The per-point pair blocks
/// come from the nuclear-attraction builder with a unit charge at the point
/// (internal/md_one_electron.hpp); the builder's block carries the
/// nuclear-attraction sign, so the potential is the positive contraction of
/// the density with those blocks. Canonical pairs contract with the mirrored
/// density (D_uv + D_vu) over the off-diagonal blocks, plain D over the
/// diagonal shells (each block covers the full shell square there).
///
/// Working set: the build materializes the transient pair store (the same
/// Theta(N^2) class as the direct builder's, freed at return) plus one
/// value per point; the per-point pair blocks are transient scratch. The
/// point axis carries the parallelism, so the retained grid mass is
/// Theta(Npoints), never Theta(Npairs x Npoints).
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell must satisfy l <= 6.
/// \param density The spin-summed AO density matrix D (n x n, host-canonical
/// tensor), the HfResult::density convention.
/// \param points The {Npoints, 3} host-canonical tensor of point
/// coordinates in Bohr, row-major point order.
/// \returns The potential values in point order, or an Error
/// (kInvalidArgument for a shape mismatch, kUnimplemented for a shell
/// beyond the parser cap, kInvalidArgument when an atom has no basis
/// entry).
/// \ingroup qcx-integrals
qcx::Result<std::vector<double>> BuildElectronPotentialAtPoints(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& points);

/// What one one-electron derivative walk counted.
///
/// The pair set the walk visited, then one column per operator, so the cost of
/// the three derivative walks this call shares can be read apart: the pairs are
/// the same for all three, but the coordinate sets are not - the
/// nuclear-attraction derivative reaches every nucleus, the other two only the
/// pair's own atoms.
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-integrals
struct OneElectronGradientCounts {
    std::size_t orbitalPairs = 0; ///< Canonical orbital pairs the walk visited.
    std::size_t overlapTuples = 0; ///< Overlap derivative tuples differentiated.
    std::size_t kineticTuples = 0; ///< Kinetic-energy derivative tuples differentiated.
    std::size_t nuclearTuples = 0; ///< Nuclear-attraction derivative tuples differentiated.
};

/// The S/T/V derivative surface at a fixed density.
///
/// With X one of S, T, V and E_X(R; D) = sum_uv D_uv X_uv(R), each gradient is
/// dE_X/dR and each contraction is the value it differentiates. The density
/// matrix is held fixed, so the derivative is the skeleton contribution only
/// [Helgaker2000]; the density's own response to the displacement - the SCF's
/// business, not this walk's - is not in it.
///
/// The contractions are reported beside their gradients because they are the
/// one number that separates the two ways a gradient can be wrong: one that
/// differentiates a different contraction differs from the matrix builders
/// here, whereas one that agrees with its own contraction's finite differences
/// while the contraction does not move with it differs nowhere else.
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-integrals
struct OneElectronGradient {
    /// dE_S/dR, 3N entries in the overlap contraction's own unit per Bohr,
    /// atom-major: entry 3*i + d is direction d of atom i, the same ordering
    /// the nuclear coordinates use.
    Eigen::VectorXd overlapGradient;
    /// dE_T/dR, 3N entries in Hartree/Bohr, atom-major (the ordering above).
    Eigen::VectorXd kineticGradient;
    /// dE_V/dR, 3N entries in Hartree/Bohr, atom-major (the ordering above).
    Eigen::VectorXd nuclearGradient;
    /// The value the overlap gradient differentiates, E_S = sum_uv D_uv S_uv.
    double overlapContraction = 0.0;
    /// The kinetic energy at this density, E_T = sum_uv D_uv T_uv (Hartree).
    double kineticContraction = 0.0;
    /// The electron-nucleus energy at this density, E_V = sum_uv D_uv V_uv
    /// (Hartree).
    double nuclearContraction = 0.0;
    /// The counted cost of this assembly.
    OneElectronGradientCounts counts;
};

/// One one-electron derivative walk's settings.
/// \ingroup qcx-integrals
struct OneElectronGradientOptions {
    /// The cap on materialized pair data per chunk, in bytes. The walk keeps
    /// the geometry-only pair skeleton resident and builds one chunk of the
    /// contracted pair data at a time, releasing it before the next - the
    /// matrix builders' own chunk policy (one_electron.cpp), one value for
    /// every build-once walk. A pair heavier than the cap is its own chunk:
    /// the cap is a target, not a bound on a single pair, so zero is a valid
    /// one-pair-at-a-time walk.
    std::size_t maxPairChunkBytes = 512 * 1024 * 1024;
};

/// Assembles the nuclear-coordinate derivatives of the overlap, kinetic-energy
/// and nuclear-attraction matrices, contracted with a fixed density.
///
/// With X one of S, T, V and E_X(R; D) = sum_uv D_uv X_uv(R), the call returns
/// dE_X/dR for all three operators from ONE walk of the canonical shell pairs:
/// the pair data is built once and each pair's three derivative blocks are
/// differentiated in it.
///
/// The pair blocks come from the derivative tier of internal/md_derivative.hpp.
/// The overlap and kinetic derivative operators act only on the centre
/// carrying their shell, so their requests name the pair's two atoms - a
/// nucleus carrying neither shell contributes exactly zero - while the
/// nuclear-attraction operator's centre moves with every nucleus, so its
/// request names all 3N coordinates. The density enters the value and the
/// derivative blocks the same way, and no matrix is formed: the walk
/// accumulates the 3N gradients directly.
///
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell must satisfy l <= 6 (the parser cap).
/// \param density The density matrix, nBasis x nBasis, in the basis set's
/// function order. An off-diagonal shell pair's block covers one triangle of
/// the symmetric matrix, so its contraction reads the mirrored density too.
/// \param options The walk's settings; maxPairChunkBytes bounds the pair data
/// materialized at once.
/// \returns The three gradients, the three contractions they differentiate and
/// the counted cost, or an Error: kInvalidArgument for a density whose shape
/// does not match the basis set, kUnimplemented for a shell beyond the parser
/// cap, and the shell-pair and pair-data builders' own errors otherwise.
/// \ingroup qcx-integrals
qcx::Result<OneElectronGradient> ComputeOneElectronGradient(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    const OneElectronGradientOptions& options = {});

} // namespace qcx::integrals
