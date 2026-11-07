#pragma once

/// \file
/// The total analytic nuclear gradient at a converged closed-shell reference:
/// the electronic terms, the nuclear repulsion and the dispersion correction,
/// kept apart in the result so a consumer can read each one.
///
/// The electronic gradient is the density's skeleton terms plus the orbital
/// response's own contribution, `dE_elec/dR = dTr(D h)/dR + dE_J/dR -
/// (1/2) dE_K/dR + Tr(F D(1))` for a Hartree-Fock reference, with the
/// exchange scaled by the functional's exact-exchange fraction and the
/// exchange-correlation skeleton `dE_xc/dR` added for a restricted Kohn-Sham
/// one. Every derivative integral comes from the integrals module's
/// derivative tier, the grid terms from the grid module's own gradient walks,
/// and the response is solved by the response module's state, so this module
/// adds the term algebra and nothing else. The one thing it forms itself is
/// the nuclear-repulsion derivative: `molecule` exposes the repulsion ENERGY
/// and no derivative of it, and the pair sum differentiates in closed form.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/dispersion/dispersion.hpp"
#include "qcx/error.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/response/response_kernel.hpp"
#include "qcx/response/response_solver.hpp"
#include "qcx/response/response_state.hpp"
#include "qcx/scf/gradient_reference.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace qcx::properties {

/// The total gradient's terms, each a 3N vector in Hartree/Bohr,
/// atom-major: entry `3 * a + d` differentiates with respect to the `d`-th
/// Cartesian component of atom `a`, the ordering the nuclear coordinates use.
///
/// One vector per term rather than a sum and a note, because the failure a
/// gradient assembly can hide is a term that is silently absent: a term that
/// can be read on its own can be checked on its own.
/// Public aggregate: the fields are the API (aggregate-struct exemption, G3).
/// \ingroup qcx-properties
struct TotalGradientTerms {
    /// `d Tr(D h)/dR` at the fixed reference density, `h = T + V`.
    Eigen::VectorXd oneElectron;
    /// `dE_nuc/dR`, the analytic pair sum.
    Eigen::VectorXd nuclearRepulsion;
    /// `dE_J/dR`, `E_J = 1/2 sum D D (uv|ls)`.
    Eigen::VectorXd coulomb;
    /// `-(1/2) c_HF dE_K/dR`, `E_K = 1/2 sum D D (ul|vs)` - the exchange as the
    /// closed-shell energy subtracts it, scaled by the exact-exchange fraction
    /// the functional carries (c_HF = 1 on the Hartree-Fock route).
    Eigen::VectorXd exchange;
    /// `dE_xc/dR` at the fixed reference density, the exchange-correlation
    /// skeleton term. Empty on the Hartree-Fock route, where no functional is
    /// integrated; a reference that integrates one always has it, so an empty
    /// vector means "not a Kohn-Sham calculation", never a computed zero.
    Eigen::VectorXd xc;
    /// The energy-weighted response term `Tr(F D(1))`, one entry per
    /// displacement.
    Eigen::VectorXd response;
    /// `dE_disp/dR`; empty when no dispersion correction was requested (an
    /// absent term stays absent rather than being a zero that a consumer
    /// could read as a computed one).
    Eigen::VectorXd dispersion;
};

/// The energies the gradient's terms belong to, in Hartree.
/// Public aggregate: the fields are the API (aggregate-struct exemption, G3).
/// \ingroup qcx-properties
struct TotalGradientEnergies {
    /// The reference's own electronic energy, `1/2 Tr[D (H + F[D])]`, as the
    /// SCF reported it.
    double electronic = 0.0;
    /// The electronic energy the derivative walks' contractions rebuild -
    /// `Tr(DT) + Tr(DV) + E_J - (1/2) c_HF E_K`, plus `E_xc` when a density
    /// functional is integrated. It exists to be compared with `electronic`:
    /// the residual is one of the gates this entry refuses on.
    double reconstructedElectronic = 0.0;
    /// The nuclear repulsion at the reference geometry.
    double nuclearRepulsion = 0.0;
    /// The dispersion correction; zero when none was requested.
    double dispersion = 0.0;
    /// `electronic + nuclearRepulsion + dispersion`.
    double total = 0.0;
};

/// What the underlying walks counted, kept so the cost of a gradient stays
/// readable.
/// Public aggregate: the fields are the API (aggregate-struct exemption, G3).
/// \ingroup qcx-properties
struct TotalGradientCounts {
    /// The one-electron derivative walk's counts.
    integrals::OneElectronGradientCounts oneElectron;
    /// The two-electron derivative Fock-block walk's counts.
    integrals::TwoElectronFockDerivativeCounts twoElectron;
};

/// A converged reference's total analytic gradient, term by term.
/// Public aggregate: the fields are the API (aggregate-struct exemption, G3).
/// \ingroup qcx-properties
struct TotalGradient {
    /// The total gradient: the terms summed in the order oneElectron,
    /// nuclearRepulsion, coulomb, exchange, xc (when a density functional is
    /// integrated), response, dispersion. It is that sum and not an
    /// independently assembled number, so the decomposition and the total can
    /// never disagree.
    Eigen::VectorXd gradient;
    /// The decomposition.
    TotalGradientTerms terms;
    /// The energies the gradient belongs to.
    TotalGradientEnergies energies;
    /// The walks' counted cost.
    TotalGradientCounts counts;
    /// Atoms in the reference molecule.
    std::size_t atomCount = 0;
    /// Basis functions.
    std::size_t numBasisFunctions = 0;
    /// Occupied spatial orbitals.
    std::size_t numOccupied = 0;
    /// Virtual spatial orbitals.
    std::size_t numVirtual = 0;
    /// What the response solves cost; one cold solve per coordinate.
    response::ResponseStateReport response;
    /// The reference's stationarity measure, `max|2 F_ai|` in the MO basis.
    /// The entry refuses when it exceeds TotalGradientOptions'
    /// stationarityTolerance.
    double stationarityResidual = 0.0;
    /// `|reconstructedElectronic - electronic|`. The entry refuses when it
    /// exceeds TotalGradientOptions' energyTolerance.
    double energyResidual = 0.0;
    /// The names of the right-hand-side contributions the response solves
    /// read, in the order they were wired in. A missing contribution is
    /// invisible in the total; this list is what makes it visible.
    std::vector<std::string> rightHandSideTerms;
};

/// The exchange-correlation inputs a restricted Kohn-Sham total gradient
/// needs.
///
/// A Kohn-Sham reference's energy carries `E_xc[D]` and its Fock carries the
/// potential `V_xc[D]`. The gradient differentiates both at the reference
/// density, and both have to be formed on the grid the SCF integrated over:
/// a walk that re-derived the grid settings or the screening tolerance would
/// be the derivative of a different energy, and every number in it would
/// still look reasonable. These are therefore the run's own keys, resolved by
/// the caller exactly as the energy path resolved them.
/// Public aggregate: the fields are the API (aggregate-struct exemption, G3).
/// \ingroup qcx-properties
struct TotalGradientXcOptions {
    /// A shipped excgrid functional name (excgrid::FunctionalNames()). The
    /// exact-exchange fraction is read off it, the same way the SCF seam read
    /// it.
    std::string functional;
    /// The grid settings the SCF's engine was created with.
    grid::XcGridSettings grid;
    /// The screening tolerance the energy path passed to the engine: the
    /// potential, the gradient walk and the kernel action all screen at it.
    double screeningTolerance = 0.0;
    /// The batch granularity the energy path was handed. It moves no number
    /// but the summation order; the walk's energy is compared with the energy
    /// path's, so the two walks take it from the same place.
    grid::XcBatchSettings batch;
};

/// Total-gradient settings.
/// Public aggregate: the fields are the API (aggregate-struct exemption, G3).
/// \ingroup qcx-properties
struct TotalGradientOptions {
    /// The dense ERI tensor's build settings. `accuracy` is the ONE screening
    /// preset of the calculation - it is read by the tensor build and by the
    /// two-electron derivative walk (the one-electron walk forms no pair
    /// screen) - and `maxTensorBytes` is the admission cap of the reference
    /// path (a dense tensor is `O(n^4)` by construction).
    ///
    /// On the matrix-free band the same settings feed the derivative's direct
    /// Fock builder: `accuracy` sets its Schwarz threshold, `screen` its
    /// density and per-element screening, and `maxBatchBytes` its batch cap.
    /// The builder itself reads no `maxTensorBytes`, because nothing dense is
    /// built; the automatic band policy does read it, comparing the cap with
    /// the dense estimate to choose the band at all.
    integrals::EriDenseOptions eri;
    /// Whether the two-electron derivative walk screens its quartets. Off is
    /// the exact contraction and costs accordingly.
    bool useDensityScreening = true;
    /// Which representation the response operator reads its two-electron term
    /// from.
    ///
    /// kAuto (the default) resolves at state build through
    /// ResolveOperatorBand: the dense tensor when its a-priori estimate fits
    /// `eri.maxTensorBytes`, the matrix-free band when it does not. The dense
    /// band is the faster per-application representation wherever the tensor
    /// fits - its operator is a precomputed contraction - and the matrix-free
    /// band is the one that runs past the cap, at one direct Fock build per
    /// application. Both bands remain explicitly selectable: kDenseTensor is
    /// the agreement test's reference, kMatrixFree the fallback. The
    /// matrix-free application's factor-of-three per-application target is
    /// unmet by construction (matrix_free_eri_derivative.hpp records why and
    /// where the timed comparison is made); production does not need it where
    /// the tensor fits.
    response::ResponseOperatorBand operatorBand = response::ResponseOperatorBand::kAuto;
    /// The per-coordinate response solvers' settings. One solver is kept per
    /// displacement inside the response state.
    response::ResponseSolveOptions solver;
    /// The exchange-correlation correction. Empty is the Hartree-Fock route:
    /// no functional is integrated and `TotalGradientTerms::xc` stays empty.
    /// Present, the reference is a restricted Kohn-Sham one and the entry
    /// evaluates the potential and the exchange-correlation energy on this
    /// grid, reconstructs the Kohn-Sham Fock the response reads, and adds the
    /// exchange-correlation skeleton term.
    std::optional<TotalGradientXcOptions> xc;
    /// The dispersion correction to add. Empty adds no term and leaves
    /// `TotalGradientTerms::dispersion` empty. A D4 request takes the
    /// dispersion module's frozen-charge tier; that entry's refusals (a
    /// charged molecule, an uncovered element) are propagated.
    std::optional<dispersion::ResolvedDispersion> dispersion;
    /// The gate on `max|2 F_ai|`: a reference above it is not a stationary
    /// point of the energy the gradient differentiates, so its gradient is
    /// refused before it is emitted.
    double stationarityTolerance = 1e-6;
    /// The gate on `|E_reconstructed - E_reference|`: the derivative walks'
    /// own energies must agree with the reference's electronic energy.
    double energyTolerance = 1e-8;
};

/// The Kohn-Sham half of the reference a response state is built for.
///
/// The Hartree-Fock route reads the Roothaan Fock as the SCF module's
/// evaluator returns it. A restricted Kohn-Sham reference's Fock is
///     `F_KS = (H + J) + V_xc - c_HF ((H + J) - F_RHF)`,
/// whose parenthesised difference is `K/2`, the exchange the functional
/// scales. `BuildResponseState` forms it from the evaluator's Coulomb-only
/// Fock, the potential below and the evaluator's full Fock, so the response
/// operator and the energy-weighted term read the matrix the reference was
/// converged against. A functional with `c_HF = 0` needs no exchange at all,
/// and the entry then never builds the full Fock.
///
/// The exchange fraction and the kernel are the two values
/// `ResponseState::Create` takes for a Kohn-Sham operator. The state copies
/// the callback but not what it captures, so the caller keeps the grid engine
/// and the kernel action the callback reads alive for as long as the state is
/// used.
/// Public aggregate: the fields are the API (aggregate-struct exemption, G3).
/// \ingroup qcx-properties
struct TotalGradientKohnShamRoute {
    /// The exact-exchange fraction `c_HF` the functional carries, in [0, 1].
    double exchangeFraction = 1.0;
    /// The restricted potential `V = (V_alpha + V_beta) / 2` at the reference
    /// density, `numBasisFunctions x numBasisFunctions`, in the AO order the
    /// reference's matrices use.
    Eigen::MatrixXd potential;
    /// The functional's kernel term for the orbital-Hessian action.
    response::KohnShamKernelTerm kernel;
};

/// Resolves the automatic operator-band policy.
///
/// The dense tensor's a-priori estimate - DenseEriTensorEstimateBytes at the
/// basis' shell-pair count, the same estimate and cap BuildEriTensorGeneral
/// checks before it enumerates anything - chooses the band: kDenseTensor when
/// the estimate is at most eriOptions.maxTensorBytes, kMatrixFree otherwise.
/// The dense band is the faster per-application representation where it fits,
/// so the automatic policy prefers it; the matrix-free band is the one that
/// runs when the tensor cannot be admitted.
/// \param molecule The reference molecule.
/// \param basisSet The basis the reference was run in.
/// \param eriOptions The dense build settings; maxTensorBytes is the active
/// admission cap.
/// \returns kDenseTensor or kMatrixFree, or the shell-pair build's error
/// (kInvalidArgument for an atom without a basis entry, kUnimplemented for a
/// shell beyond the parser cap).
/// \ingroup qcx-properties
qcx::Result<response::ResponseOperatorBand> ResolveOperatorBand(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const integrals::EriDenseOptions& eriOptions = {});

/// Builds the response state a total-gradient calculation solves against.
///
/// The state owns copies of everything its operator reads, so the heavy work
/// happens here once. The band is kAuto by default: the entry resolves it
/// through ResolveOperatorBand first, so a run whose dense tensor is admitted
/// takes the dense band and a run past the cap takes the matrix-free one. On
/// the dense-tensor band the AO ERI tensor is built through the dense
/// reference builder (admission-capped) and transformed to the MO basis by
/// four quarter-transforms. On the matrix-free band no tensor is built at
/// all: the entry creates a MatrixFreeEriDerivative, whose direct screened
/// Fock engine supplies both the reference Fock and every operator
/// application, and the state holds no n^4 array.
///
/// On either band the AO Fock is evaluated at the reference density by the
/// module that owns the contraction - the SCF module's own evaluator
/// (`EvaluateRhfDensityEnergy` - the Roothaan Fock, not a second integral
/// path) on the dense band, the derivative's builder on the matrix-free one,
/// which runs the same closed-shell algebra - and the screening structure is
/// stamped with the reference geometry's digest.
///
/// On the Kohn-Sham route the evaluator's Fock is not the reference's: the
/// entry rebuilds it from the Coulomb-only Fock, the evaluated potential and,
/// when the functional carries exact exchange, the full Fock, in the algebra
/// TotalGradientKohnShamRoute documents. The rebuilt matrix is what the state
/// owns and what the response operator and the energy-weighted term read.
///
/// The screening structure is at AO-pair granularity, one entry per pair
/// `mu <= nu`, because that is the shape the response state declares. Each
/// AO pair carries the bound of its containing shell pair, which is the
/// shell pair's own maximum over the block - the conservative direction. The
/// state carries and guards these bounds; it does not interpret them.
///
/// The reference is re-validated here for what this entry reads: its arrays
/// must be square and shaped by the same basis-function count the basis set
/// gives, the molecule must be closed shell, and its electron count must be
/// twice the stated occupied count. The state's own factory then refuses a
/// non-converged reference, a density that is not `2 C_occ C_occ^T` for the
/// stated occupied count, an asymmetric Fock, and a screening digest from
/// another geometry.
/// \param molecule The reference molecule, coordinates in Bohr.
/// \param basisSet The basis set the reference was run in.
/// \param reference The converged SCF state to differentiate (scf::
/// ExtractGradientReference).
/// \param eriOptions The dense tensor's build settings, and on the
/// matrix-free band the direct builder's screen and batch settings.
/// \param solverOptions The settings every per-coordinate solver starts with.
/// \param kohnSham The Kohn-Sham route's potential, exchange fraction and
/// kernel term. Absent (the default), the reference's own Fock is the
/// operator's and the state is the Hartree-Fock one; present, the Fock is
/// reconstructed and the state carries the Kohn-Sham operator. The kernel
/// callback's capture must outlive the returned state.
/// \param band Which representation the operator reads its two-electron term
/// from. kAuto (the default) resolves through ResolveOperatorBand; kMatrixFree
/// builds no dense tensor; kDenseTensor is the admission-capped tensor path,
/// kept reachable for the agreement test.
/// \returns The state, or an Error: kInvalidArgument naming the shape, count
/// or consistency refusal; kUnimplemented for a non-closed-shell molecule;
/// the tensor build's own kOutOfMemory or admission refusal (dense band) or
/// the direct builder's refusal (matrix-free band); the response module's
/// refusal otherwise.
/// \ingroup qcx-properties
qcx::Result<response::ResponseState> BuildResponseState(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::scf::ScfGradientReference& reference,
    const integrals::EriDenseOptions& eriOptions = {},
    const response::ResponseSolveOptions& solverOptions = {},
    std::optional<TotalGradientKohnShamRoute> kohnSham = std::nullopt,
    response::ResponseOperatorBand band = response::ResponseOperatorBand::kAuto);

/// Assembles the total analytic gradient at a converged closed-shell
/// reference.
///
/// The terms, in the order the result's gradient accumulates them:
///
/// 1. `dTr(D h)/dR` from the one-electron derivative walk;
/// 2. `dE_nuc/dR`, the analytic pair sum;
/// 3. `dE_J/dR` and `-(1/2) c_HF dE_K/dR` from the two-electron derivative
///    Fock walk (one call covers every displacement; its skeletons are the
///    same contractions its blocks differentiate), `c_HF = 1` on the
///    Hartree-Fock route and the functional's exact-exchange fraction on the
///    Kohn-Sham one;
/// 4. `dE_xc/dR` when a density functional is integrated, from the grid
///    module's exchange-correlation gradient walk at the fixed reference
///    density;
/// 5. `Tr(F D(1))` per displacement, the orbital response solved by the
///    response state against the right-hand side
///    `H'_ov + J'_ov - (1/2) c_HF K'_ov + V'_xc,ov` and the overlap
///    derivative `C^T S' C`;
/// 6. `dE_disp/dR` when a dispersion correction was requested.
///
/// Two gates run before anything is emitted, because both failures produce a
/// plausible-looking gradient from a reference the gradient does not belong
/// to: the reference must be stationary (`max|2 F_ai|` within the option,
/// against the Kohn-Sham Fock when a functional is integrated - the matrix
/// the SCF that reported the reference actually converged) and the derivative
/// walks' own energies must reproduce the reference's electronic energy
/// (within the option; the reconstruction adds `E_xc` and subtracts
/// `(1/2) c_HF E_K` on the Kohn-Sham route). A refused call returns no
/// partial result - the result carries no gradient until both gates have
/// passed.
/// \param molecule The reference molecule (its coordinates are the geometry
/// the gradient is taken at).
/// \param basisSet The basis set the reference was run in.
/// \param reference The converged SCF state to differentiate.
/// \param options The build, screening, solver, exchange-correlation,
/// dispersion and gate settings.
/// \returns The gradient with its decomposition, energies, counts and solve
/// report, or an Error: the response state's refusals, the exchange-
/// correlation engine's or walks' own errors, the derivative walks' own
/// errors, the stationarity or energy-reconstruction refusal
/// (kInvalidArgument, naming the residual and the tolerance), or the
/// dispersion entry's refusal (kUnimplemented for a charged D4 molecule or an
/// element its table does not cover).
/// \ingroup qcx-properties
qcx::Result<TotalGradient> ComputeTotalGradient(const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& basisSet,
                                                const qcx::scf::ScfGradientReference& reference,
                                                const TotalGradientOptions& options = {});

} // namespace qcx::properties
