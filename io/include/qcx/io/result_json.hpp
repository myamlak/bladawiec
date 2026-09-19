// The qcx run result schema (the properties block included): one JSON
// document per run, serialized by
// SerializeRunResultJson. The schema carries exactly what the code can
// honestly emit: convergence, energies, spin-squared (UHF runs only - RHF
// emits null, never a fabricated diagnostic zero), the population/moment
// block, and wall times. Optional members emit null when
// unset - never placeholder numbers that would read as real output to a
// consumer of the JSON. Non-finite numbers (NaN, +/-Infinity) serialize as
// the explicit string markers "nan"/"inf"/"-inf" - never as null, which is
// reserved for unset members.

#pragma once

#include "qcx/error.hpp"
#include "qcx/io/run_input.hpp"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace qcx::io {

/// The population block: the four analyses of
/// qcx::properties, per atom in the molecule's canonical atom order. All
/// values are in electrons; the per-spin arrays follow the properties
/// module's convention (RHF: alpha == beta, both D/2).
/// \ingroup qcx-io
struct RunPopulations {
    std::vector<double> mullikenAlpha; ///< Per-atom gross alpha population.
    std::vector<double> mullikenBeta; ///< Per-atom gross beta population.
    std::vector<double> mullikenTotal; ///< alpha + beta.
    std::vector<double> mullikenSpin; ///< alpha - beta (zero for RHF by construction).
    std::vector<double> lowdinAlpha; ///< Per-atom Lowdin alpha population.
    std::vector<double> lowdinBeta; ///< Per-atom Lowdin beta population.
    std::vector<double> lowdinTotal; ///< alpha + beta.
    std::vector<double> lowdinSpin; ///< alpha - beta.
    std::vector<std::vector<double>>
        mayerBondOrders; ///< nAtoms x nAtoms, symmetric, zero diagonal.
    std::vector<double> mayerFreeValences; ///< Zero for RHF by construction.
    std::vector<double> mayerTotalValences; ///< Free valence plus bond-order row sum.
    std::vector<std::vector<double>>
        gopinathanJugBondOrders; ///< nAtoms x nAtoms, symmetric, diagonal included.
};

/// The electric moments, relative to the run input's
/// coordinate origin as given (the caller recenters the molecule first if
/// a center-of-mass dipole is wanted).
/// \ingroup qcx-io
struct RunMoments {
    std::array<double, 3> dipole = {}; ///< (d_x, d_y, d_z), e*a0.
    std::array<std::array<double, 3>, 3> quadrupole =
        {}; ///< Traceless Theta = 3M - Tr(M) I, e*a0^2, symmetric.
};

/// The charge block: the Hirshfeld/Voronoi charges of
/// the opt-in [properties] block, per atom in the molecule's canonical
/// atom order, in electrons. Each member is optional: the serializer emits
/// null for an analysis that was not requested (never a fabricated zero).
/// \ingroup qcx-io
struct RunCharges {
    std::optional<std::vector<double>> hirshfeld; ///< [Hirshfeld1977] stockholder charges.
    std::optional<std::vector<double>> voronoi; ///< [FonsecaGuerra2004] cell charges.
};

/// One ESP point-charge fit.
/// \ingroup qcx-io
struct RunEspFit {
    std::vector<double> charges; ///< Per-atom charges, molecule atom order, electrons.
    double rmsError = 0.0; ///< RMS fit deviation over the point set, hartree.
    std::size_t pointCount = 0; ///< Fit points actually used.
};

/// The EDDB delocalized-bond analysis, spinless populations in
/// electrons. The working-basis density matrix is not serialized (it lives
/// in the orthogonalized representation, meaningless to a JSON consumer);
/// the descriptors are the populations and the NOBD occupations.
/// \ingroup qcx-io
struct RunEddb {
    double totalPopulation = 0.0; ///< N_EDDB = Tr D~_EDDB.
    std::vector<double> atomicPopulations; ///< Per-atom delocalized populations.
    std::vector<double> nobdOccupations; ///< Delocalization-orbital occupations, descending.
    std::size_t centralAtomCount = 0; ///< Atoms with >= 2 selected neighbors.
    std::size_t twoCenterOrbitalCount = 0; ///< Retained two-center bond-order orbitals.
};

/// The condensed Fukui indices, per atom, each vector summing
/// to one.
/// \ingroup qcx-io
struct RunFukui {
    std::vector<double> nucleophilic; ///< f_A^+.
    std::vector<double> electrophilic; ///< f_A^-.
    std::vector<double> radical; ///< f_A^0 = (f^+ + f^-) / 2.
};

/// The Nalewajski-Mrozek bond orders: the Scheme-III bond orders
/// and their building blocks, symmetric zero-diagonal matrices in the
/// canonical atom order (electrons).
/// \ingroup qcx-io
struct RunNalewajski {
    std::vector<std::vector<double>> bondOrders; ///< B(a,b).
    std::vector<std::vector<double>> diatomicCovalent; ///< v_ab.
    std::vector<double> atomicIonicValence; ///< v_i_a.
    std::vector<double> atomicCovalentValence; ///< v_c_a.
    std::vector<double> totalValence; ///< total_ab(a).
};

/// The ETS-NOCV energy decomposition: the fragment-based binding
/// energy split into E_elstat + E_Pauli + E_orb, with the sign-paired
/// channel resolutions, all in Hartree except the channel
/// eigenvalues (electrons) and the fragment energies (Hartree). The
/// density matrices of the decomposition are not serialized - the energies
/// and channels are the output.
/// \ingroup qcx-io
struct RunNocv {
    double electrostatic = 0.0; ///< E_elstat.
    double pauli = 0.0; ///< E_Pauli.
    double orbital = 0.0; ///< E_orb, restricted functional.
    double orbitalUnrestricted = 0.0; ///< E_orb, spin-resolved.
    double bindingEnergy = 0.0; ///< E_mol - sum_i E_frag_i = E_elstat + E_Pauli + E_orb.
    std::vector<double> fragmentEnergies; ///< Isolated fragment SCF energies, fragment order.
    std::vector<double> orbitalComponents; ///< Delta E_orb^k per sign-paired channel.
    std::vector<double> nocvEigenvalues; ///< The +nu_k of each pair (signed tail).
    std::vector<double> orbitalComponentsAlpha; ///< Alpha-channel resolution.
    std::vector<double> orbitalComponentsBeta; ///< Beta-channel resolution.
    std::vector<double> nocvEigenvaluesAlpha; ///< Alpha-channel nu_k.
    std::vector<double> nocvEigenvaluesBeta; ///< Beta-channel nu_k.
};

/// The electron density at each nucleus: the spin-summed AO
/// density evaluated at every nucleus, electrons/bohr^3.
/// \ingroup qcx-io
struct RunDensityAtNuclei {
    std::vector<double> values; ///< rho(R_A) per atom, molecule atom order.
};

/// One (3,-1) bond critical point of the Bader QTAIM block: a saddle of
/// rho with exactly one positive Hessian eigenvalue, and the bond path to
/// its two bonded nuclei.  The properties module's BondCriticalPoint as
/// plain data, so io stays Eigen-free.
/// \ingroup qcx-io
struct RunBondCriticalPoint {
    std::size_t atomA; ///< The first bonded nucleus, molecule order.
    std::size_t atomB; ///< The second bonded nucleus, molecule order.
    std::vector<double> positionBohr; ///< 3; the critical point, Bohr.
    double density = 0.0; ///< rho at the critical point, e/bohr^3.
    double laplacian = 0.0; ///< lambda1 + lambda2 + lambda3, e/bohr^5.
    double ellipticity = 0.0; ///< lambda1/lambda2 - 1 >= 0.
    /// 3, ascending: lambda1 <= lambda2 < 0 < lambda3, e/bohr^5.
    std::vector<double> eigenvalues;
    /// The bond path: the two gradient rays BCP -> nucleus concatenated,
    /// each seeded 1e-3 bohr (kPathStartEpsilon) off the BCP along the bond
    /// direction (the BCP itself opens both rays and appears twice),
    /// endpoints snapped to the nuclei, Bohr.
    std::vector<std::vector<double>> bondPath;
};

/// A converged critical point that is not a (3,-1) bond critical point, or
/// a bond critical point whose bond path did not terminate at a nucleus.
/// Reported, never a failure (the failed-search policy).
/// \ingroup qcx-io
struct RunOtherCriticalPoint {
    std::vector<double> positionBohr; ///< 3; the critical point, Bohr.
    int rank = 3; ///< The Hessian rank.
    int signatureSum = 0; ///< The Hessian eigenvalue-sign sum (-1 | +1 | +3).
};

/// The Bader QTAIM block: the (3,-1) bond
/// critical points of rho with their bond paths, and the reported
/// non-bond / truncated-path critical points and unconverged Newton seeds.
/// \ingroup qcx-io
struct RunQtaim {
    /// The (3,-1) bond critical points, each with its bond path.
    std::vector<RunBondCriticalPoint> bondCriticalPoints;
    /// Converged non-bond critical points and bond critical points whose
    /// bond path did not terminate at a nucleus (the failed-search policy).
    std::vector<RunOtherCriticalPoint> otherCriticalPoints;
    /// 3 each; the pair-seed midpoints that did not converge, Bohr.
    std::vector<std::vector<double>> unconvergedSeeds;
};

/// The Molden export of one run: the path the
/// [Molden Format] [Atoms] (AU) [5D] [7F] [GTO] [MO] file was written to.
/// \ingroup qcx-io
struct RunMolden {
    std::string file; ///< The written file's path, as given in the input.
};

/// The properties of one run, plain data so io stays free of Eigen (the
/// driver converts from the properties module's matrix types). The
/// population/moment block is always computed; the charge blocks
/// appear only when the run requested the analysis.
/// \ingroup qcx-io
struct RunProperties {
    RunPopulations populations; ///< The population block.
    RunMoments moments; ///< The electric moments.
    std::optional<RunCharges> charges; ///< Hirshfeld/Voronoi, opt-in.
    std::optional<RunEspFit> esp; ///< ESP fit, opt-in.
    std::optional<RunEddb> eddb; ///< EDDB, opt-in.
    std::optional<RunFukui> fukui; ///< Condensed Fukui, opt-in.
    std::optional<RunNalewajski> nalewajski; ///< Nalewajski-Mrozek, opt-in.
    std::optional<RunNocv> nocv; ///< ETS-NOCV, opt-in.
    std::optional<RunDensityAtNuclei> densityAtNuclei; ///< Density at nuclei, opt-in.
    std::optional<RunQtaim> qtaim; ///< Bader QTAIM, opt-in.
    std::optional<RunMolden> molden; ///< Molden export, opt-in.
};

/// One score row of the coefficients record: a candidate's MAD score
/// with the source of that score. RETAINED SHAPE, NO PRODUCER (schema 11):
/// the candidate-ranking heuristic that wrote these rows was retired
/// by the lean flip, so nothing in a real run constructs one. The
/// shape stays because the coefficients record is versioned by its
/// `file_version` - dropping a documented member is a version bump, not an
/// edit - but a row's presence is not evidence that a ranking happened.
/// \ingroup qcx-io
struct RunSelectionScore {
    std::string candidate; ///< The candidate's short name ("direct",
                           ///< "ri_j_link", "ri_jk", "qfmm", "gpu",
                           ///< "gpu_split").
    double scoreSeconds = 0.0; ///< The row's per-Fock-build estimate.
    std::string source; ///< Where the score came from: "analytic" (the
                        ///< analytic heuristic - the unconditional
                        ///< per-candidate fallback), "table" (a measured
                        ///< coefficient cell) or "borrowed" (a borrowed
                        ///< cell: the donor's score times the stored
                        ///< inflation).
    bool selectable = false; ///< The row's admission verdict.
};

/// The coefficient-seam provenance inside the selection record (the
/// engine-seam consult answer; io stays pure). RETAINED SHAPE, NO
/// PRODUCER (schema 11): the driver's resolution layer used to fill
/// this from its candidate ranking, and that ranking was retired by the
/// lean flip - the seam's own types have no caller outside their
/// serializer test, and nothing in a real run populates the block. The
/// decision it used to carry now lives in picked/builderMember/reasoning.
/// cellKey/borrowed/inflation describe the best selectable candidate's
/// matched cell (the seam's pick; what an override diverges FROM): the
/// full canonical cell key when its score came from a cell, the borrowed
/// flag, and the inflation applied exactly when borrowed.
/// \ingroup qcx-io
struct RunSelectionCoefficients {
    std::string fileVersion; ///< The loaded coefficient file's file_version.
                             ///< No driver path loads one, so the whole
                             ///< block is absent in every run today (the
                             ///< "none" of the record).
    std::string machineClassKey; ///< The run's machine-class key (the fit
                                 ///< corpus vocabulary).
    std::string basisFamilyId; ///< The run's basis-family id.
    std::string presetId; ///< The run's method preset ("default" in v1).
    /// The best selectable candidate's matched cell's canonical key; absent
    /// when its score fell back to the analytic heuristic.
    std::optional<std::string> cellKey;
    /// True when the best selectable candidate's score came from a borrowed
    /// cell (the synthetic new-family placeholder, donor times inflation).
    bool borrowed = false;
    /// The inflation applied to the donor score; present exactly when
    /// borrowed.
    std::optional<double> inflation;
    /// The ranked score list (see the struct note); EMPTY whenever the
    /// block is emitted, because no producer exists.
    std::vector<RunSelectionScore> scores;
};

/// The selection record inside resources_resolved: the builder the run
/// wired and how it was chosen. builder keeps the FAMILY word (schema 15
/// - a lean run still reports "direct"), and
/// builderMember names the within-family member the run actually wired,
/// so the member is read off the record rather than inferred from
/// explicitBuilder's absence. explicitBuilder is present when the input
/// named a fock_builder FAMILY word; absent means no family word was
/// given. The lean carve-out: the explicit `fock_builder = "lean"`
/// spelling is a within-family request, not a family word (the kGpuSplit
/// precedent - RunMethodInput::leanDirect), so it leaves explicitBuilder
/// ABSENT by design, exactly as an omitted key does; the request is
/// recorded in reasoning, the choice it makes in builderMember.
/// warning names the divergence when the explicit choice was overridden
/// (the device-less gpu fallback), and is omitted when the run was
/// silent. coefficients (schema 11, the engine seam) carries the
/// measured-cost table's provenance of the retired heuristic's ranking;
/// absent when no coefficient file was loaded for the run.
/// \ingroup qcx-io
/// The measured per-atom error of the approximated exchange, with the bar it is
/// read against (schema 31, the approximated-exchange disclosure plumbing).
///
/// **Why a record needs it.** The rule for this path is that the
/// path closes with its error DISCLOSED - and a disclosure a run cannot emit is
/// not a disclosure. Before this key the achieved error lived only in a test
/// cell's output and in prose; the mapping
/// beside it carried the BAR and no achieved value at all.
/// So a reader of an energy built through an auxiliary fit had the aux's NAME
/// and nothing to size the error with, and the bar it misses
/// (`integrals RiExchangeErrorBudgetPerAtom`) was not visible either.
///
/// **The two numbers are of different kinds, and the record says which is
/// which.** `barPerAtomHartree` is the run's OWN: the preset's per-atom budget,
/// computed from the preset this run resolved (`barPreset` names it, because the
/// record echoes `[method] accuracy` nowhere else and a bar a reader cannot tie
/// to a preset is the shape this record forbids). `measuredPerAtomHartree` is NOT the run's -
/// no run can produce it. It is the deviation of fitted exchange from exact
/// screened exchange at the same preset, measured in the ri_jk accuracy
/// cell, and the record names the fixture and the fit it was taken on
/// (`measuredOn`, `measuredAuxBasis`) so it can never be misread as a claim
/// about this run. It is the worst of the characterized cells, which is a
/// measurement and NOT a bound - see
/// `integrals RiExchangeWorstMeasuredPerAtomError`.
///
/// **The block is the disclosure's home rather than a new sibling of it** (one
/// fact, one home): this is the quantitative half of what `exchange` states
/// qualitatively, so it hangs off the block a reader is already looking at.
/// \ingroup qcx-io
struct RunExchangeError {
    /// The achieved per-atom exchange error measured in the accuracy cell, Eh
    /// per atom - the path's characterized value, never this run's own error
    /// (see the type's note; the fixture travels in `measuredOn`).
    double measuredPerAtomHartree = 0.0;
    /// The per-atom bar the run is read against: the preset's own
    /// `RiExchangeErrorBudgetPerAtom`, for the preset `barPreset` names.
    double barPerAtomHartree = 0.0;
    /// The preset word the bar belongs to ("kLoose" | "kNormal" | "kTight"),
    /// the input's own vocabulary. Present because the record names no preset
    /// anywhere else, so without it the bar would be an unattributable number.
    std::string barPreset;
    /// The fixture the measured value was taken on, in the accuracy cell's own
    /// label vocabulary (`h2o_sto3g`, `water_def2svp`): the scope of the number
    /// beside it, which is what keeps a path measurement from reading as this
    /// run's.
    std::string measuredOn;
    /// The auxiliary fit BOTH sides of that measurement used.
    std::string measuredAuxBasis;
};

/// The exchange-approximation disclosure (schema 24). A run whose exchange half
/// is not the exact two-electron kernel must say so in its own record, or a
/// consumer reads an energy built from an auxiliary fit as though it came from
/// exact quartets - the different-result-class problem, answered by
/// DISCLOSURE rather than by deletion.
///
/// **PRESENT when EITHER an approximate builder was wired OR the auxiliary
/// basis selection carries a notice** (the notice
/// regions are `AuxSelectionNotice`'s - minimal bases, bases without
/// polarization, diffuse-augmented bases, ECP/relativistic bases, and a JK
/// request with no matched JK fit). Absent only when NEITHER holds, and that
/// absence is the null honesty policy - the "exact and no notice" of this key,
/// never a fabricated default (the certified-bound-zero defect again).
///
/// **The sentence this replaces read "Present exactly when the run wired such a
/// builder; absent on every run whose kernels are exact", and that premise is
/// FALSE.** The reason is the recurring defect class: a notice
/// hanging off an ABSENT block is "calculated and invisible", and a reader who
/// sees a warning must not have to infer that the block was omitted because the
/// kernel was exact. If the kernel IS exact while a notice exists, that is itself
/// information the reader needs - stated here, where they are already looking,
/// with `exchange` carrying it (see that field).
/// \ingroup qcx-io
struct RunApproximation {
    /// The contraction form the exchange was built in - the member word, so the
    /// disclosure and `builder_member` name the same thing. On a notice-only
    /// block (an exact kernel with a warning) this reads "exact" - the
    /// reasoning being the struct's own: its
    /// history makes "exact" the meaning of the ABSENT case, so a present block
    /// must state what the reader would otherwise have had to reconstruct.
    std::string exchange;
    /// The auxiliary basis name IN EFFECT for that contraction: the explicit
    /// [basis].aux when the input carried one, else the resolved auto-selection.
    /// It is recorded as the aux the RUN used, never as a claim about the input
    /// (the auto-selected default must not read back as a user request).
    std::string auxBasis;
    /// Why this auxiliary basis was chosen by a policy that WARNS here, when it
    /// was; absent when the selection is in no weak region.
    ///
    /// This is the disclosure half of the demotion outcome. The aux rule
    /// always produces a default, so a run in a weak region is demoted-to-a-
    /// default WITH disclosure rather than refused - and a demotion the record
    /// cannot show is not a demotion, it is the silent substitution the contract
    /// exists to forbid. `optional` rather than a present-but-empty string, so
    /// that absence means "no notice" and never "empty notice", and so the
    /// common case costs the record nothing.
    std::optional<std::string> auxNotice;
    /// The measured per-atom error of the approximated exchange, beside the bar
    /// this run's preset reads it against (schema 31, see RunExchangeError).
    ///
    /// **Present exactly when `exchange` names an APPROXIMATED contraction** -
    /// never on the notice-only block whose kernel is exact. That is the whole
    /// presence rule, and it is narrower than the block's own: the block is
    /// emitted on an exact-kernel run carrying a notice (schema 25), where the
    /// fitted exchange's error is not this run's error and printing it would be
    /// a lie about the run (the exact run's own absence arm, asserted in
    /// `run_driver_test.cpp`).
    std::optional<RunExchangeError> exchangeError;
};

/// The builder-resolution record of resources_resolved: the builder the run
/// wired, the within-family member that ran, and the reason the resolution
/// gave - the OUTCOME and the REQUEST kept distinct, so a default the system
/// chose can never read back as a choice the user made.
/// \ingroup qcx-io
struct RunSelection {
    BuilderKind builder; ///< The builder the run actually wired — the FAMILY
                         ///< word ("direct" on the lean member too).
    /// The builder the run wires WITHOUT an explicit fock_builder - the
    /// no-key size ladder's own outcome (selection_resolution.hpp: the
    /// direct family's lean member at nBasis <= 1000, ri_j_link to 2000,
    /// qfmm above, and the tier a demotion landed on when the named tier
    /// could not be wired). It is NOT a single family: "the direct family
    /// default at every size" described the lean flip and stopped being
    /// true when the size ladder replaced it. The selection's
    /// "picked" field in the JSON; never a model pick - that ranking is
    /// retired.
    std::optional<BuilderKind> picked;
    /// The wired builder's within-family MEMBER name (schema 15), the
    /// additive sibling of `builder` above: "lean" when the direct family
    /// wired its within-family lean (Schwarz-only) member - the ladder's
    /// floor at nBasis <= 1000, or the explicit `fock_builder = "lean"` -
    /// "occ_ri_k" when the full-RI exchange family wired its occupied-orbital
    /// contraction form, and the family word itself on every other run. So a
    /// consumer reads which member ran instead of inferring it (an absent
    /// explicitBuilder is NOT that signal: the lean carve-out leaves it
    /// absent by design). Present on every run that reached the resolution
    /// (the `picked` contract); absent on a record that never resolved.
    std::optional<std::string> builderMember;
    /// The resolution's why, verbatim, one line. The opener names the
    /// request: "no fock_builder given: ..." for an omitted key,
    /// "explicit fock_builder <word> honored ..." for a request (the lean
    /// request included - it carries its own text, never the omitted-key
    /// opener).
    std::string reasoning;
    /// The input's explicit builder FAMILY word, when one was given (the
    /// lean spelling never lands here - see the type's own note).
    std::optional<BuilderKind> explicitBuilder;
    /// The divergence warning text; omitted when the run was silent.
    std::optional<std::string> warning;
    /// The exchange-approximation disclosure (schema 24, see RunApproximation);
    /// absent only when every kernel the run used was exact AND the aux
    /// selection carries no notice. The presence rule widened this: a notice on
    /// an exact-kernel run keeps the block PRESENT, so a warning never hangs off
    /// an absent block - see the struct's own note for why that is the defect
    /// class and not a nicety.
    std::optional<RunApproximation> approximation;
    /// The coefficient-seam provenance block; absent when no
    /// coefficient file was loaded (the seam then sat inert and every
    /// score was the analytic heuristic - absence IS the "none" of the
    /// consult answer's file-version record, never a fabricated default).
    std::optional<RunSelectionCoefficients> coefficients;
};

/// The workspace_budget record of resources_resolved: the
/// cap-minus-base workspace budget the budget path granted the engine,
/// and the cumulative commit its Create-time reservations charged.
/// Present only when the run executed on the budget path (a positive cap
/// that cleared the modeled base term): ri_j_link, the direct family's
/// adaptive seam, and the composed-QFMM runs (the one shared
/// budget reaches both nested halves, so the commit covers both
/// Creates). The gpu family and the legacy null-budget path leave it
/// absent.
/// \ingroup qcx-io
struct RunWorkspaceBudget {
    std::size_t capacityBytes = 0; ///< The cap-minus-base capacity granted.
    std::size_t committedBytes = 0; ///< The cumulative Create-time commit.
};

/// The mode_record of resources_resolved: the engine's Create-time mode
/// decision and its firing estimate terms (the fock_build.hpp FockModeInfo
/// mirrored into the schema layer — io stays pure). Present only when a
/// workspace budget was in effect (the budget path — ri_j_link, the
/// direct family's adaptive seam, and the composed-QFMM runs' Coulomb
/// (QFMM) half; the gpu family and the legacy
/// null-budget path leave it absent); the record
/// names the rung that fired, the estimate/reservation/budget bytes, and
/// the term-by-term breakdown of the fired estimate. Zero bytes mean the
/// term is not applicable to the builder's family; the two exclusion
/// flags say an a-priori exclusion fired (the light rung was mandatory).
/// The composed-QFMM runs share one budget between TWO nested builders
/// (the QFMM half and the exchange-only direct half, reservation order =
/// nesting order): mode_record carries the Coulomb half's decision, and
/// the exchange half's own decision is visible under
/// exchange_mode_record on UHF runs (both admission-gate observations
/// must be visible there). On the composed-QFMM RHF surface the record is
/// the Coulomb half's alone; the exchange half's commit is inside
/// workspace_budget's committed_bytes.
/// \ingroup qcx-io
struct RunModeRecord {
    std::string mode; ///< The rung that fired: "kFastPath", "kLightPath" or
                      ///< "kDisk" (the engine's enumerator vocabulary; kDisk is
                      ///< recorded by the driver's disk route).
    /// Whether the disk route was FORCED rather than ladder-selected (schema
    /// 18, the input's method.ri_tensor_mode = "forced_disk" diagnostic
    /// override): true only on that route, false on every engine rung
    /// decision and on the ladder's own disk fallback. It is what lets a
    /// reader tell a forced-disk measurement cell from a run the ladder
    /// actually sent to disk - and, with `mode` still naming the store
    /// ("kDisk"), it leaves the mode vocabulary the engine's own. Always
    /// present on a serialized mode record: an absent member could not
    /// distinguish the two admissions.
    bool forcedDisk = false;
    std::size_t predictedBytes = 0; ///< The full Create-time footprint estimate.
    std::size_t reservedBytes = 0; ///< Bytes actually reserved (the nested-builder
                                   ///< subtraction, RI only).
    std::size_t budgetBytes = 0; ///< The budget's capacity at Create.
    std::size_t remainingAtDecision = 0; ///< Budget remaining at the decision.
    std::size_t maxBatchBytes = 0; ///< The batch cap in force.
    std::size_t pairStoreBytes = 0; ///< MD pair data (E tables, transforms, weights).
    std::size_t patternBytes = 0; ///< Neighbor CSR indices.
    std::size_t scratchBytes = 0; ///< Per-thread batch arena.
    std::size_t structuralBytes = 0; ///< Pair list, Schwarz vector, CSR offsets, core-H copies.
    std::size_t cacheBytes = 0; ///< ERI cache (plain path only).
    std::size_t lightStoreBytes = 0; ///< LightPath geometry-only pair store.
    std::size_t chunkArenaBytes = 0; ///< LightPath peak-chunk arena.
    std::size_t chunkPatternBytes = 0; ///< LightPath peak-chunk pattern.
    std::size_t chunkIndexBytes = 0; ///< LightPath per-call chunk bookkeeping.
    std::size_t lightShellsBytes = 0; ///< LightPath retained flattened shells.
    std::size_t chunkPairs = 0; ///< LightPath chunk size in bra rows.
    std::size_t diskBytes = 0; ///< The disk rung's modeled on-disk (uv|P) payload
                               ///< (the driver's disk route; 0 elsewhere).
    std::size_t tensorBytes = 0; ///< RI-J (uv|P) values buffer (8 n^2 nAux).
    std::size_t riMatrixBytes = 0; ///< RI-J retained n^2 x nAux Eigen copy.
    std::size_t taskListBytes = 0; ///< RI-J screened task-list charge (16 B per surviving task).
    std::size_t metricBytes = 0; ///< RI-J metric store and eigendecomposition.
    std::size_t orbitalAuxBytes = 0; ///< RI-J orbital and aux pair stores.
    std::size_t outerStoreBytes = 0; ///< QFMM octree node vector.
    std::size_t exchangeBytes = 0; ///< The nested direct-exchange half's estimate.
    std::size_t exchangeScratchBytes = 0; ///< The nested half's batch arena.
    bool patternExcluded = false; ///< Exclusion (i): even the all-survive pattern
                                  ///< cannot fit — the light rung is mandatory.
    bool tensorExcluded = false; ///< Exclusion (ii, RI-J): the tensor term alone
                                 ///< cannot fit — the light rung is mandatory.
    std::size_t classTableBytes = 0; ///< The pair-class table's Create-time structural
                                     ///< charge (the class path; 0 when the path disengaged or
                                     ///< was never requested).
    bool classPathDisengaged = false; ///< The class-path admission gate fired: the table's
                                      ///< never-under estimate could not fit the budget, so the
                                      ///< plain screened path ran instead (the results are
                                      ///< bit-identical).
    /// The Create-time authorized concurrent batch slots (the
    /// bounded-concurrency batch loop's k): the engine's budgeted-decision
    /// records carry min 1 (the slots never authorized still run one);
    /// 0 only on the driver-synthesized disk record, which no engine
    /// decision produced. Mirrors FockModeInfo::concurrentSlots (the
    /// runtime per-pass kEff can sit below it - the observed per-call
    /// concurrency is a FockBuildStats read, not part of this record).
    /// The mirror was added later (the k a fired run carried was
    /// previously dropped from the JSON).
    std::size_t concurrentSlots = 0;
    /// The decision's clamp-origin team read: DefaultOmpTeamSize() at
    /// Create on the budgeted path - the team the k = min(...) decision
    /// clamped against. A run with concurrent_slots == default_team_size
    /// was team-clamped (k = the whole ceiling-bounded team); one below it
    /// was budget-clamped. 0 on the legacy no-budget path. Mirrors
    /// FockModeInfo::defaultTeamSize.
    std::size_t defaultTeamSize = 0;
};

/// The composed full-RI builder's Create-time rung record of
/// `resources_resolved` (schema 30): the ri_jk family's own memory-ladder
/// decision, mirrored from the integrals module's `RiFullFockModeInfo`. The
/// counterpart of RunModeRecord for a different ladder, and deliberately a
/// separate block rather than a member of it: the two families' rung words
/// name different rungs (`kFast`/`kBlocked` are the RI-K rungs, the batched
/// blocked one the RI-J ladder does not have) and their term decompositions
/// are different allocations, so one block carrying both would make every
/// number's meaning depend on which family wrote it.
///
/// Written only when a decision was actually made (the builder was created
/// WITH a workspace budget): the no-budget path consults no budget and takes
/// the fast rung, so the block is ABSENT there rather than reporting a
/// decision that did not happen.
/// \ingroup qcx-io
struct RunRiJkMode {
    /// The rung that fired, in the engine's enumerator vocabulary:
    /// `"kFast"` (the raw 3-center tensor and its transform live at once) or
    /// `"kBlocked"` (the transform accumulated one auxiliary shell range at a
    /// time, the ladder's batched rung at no extra flops).
    std::string rung;
    /// The engaged rung's Create-plus-first-call peak, bytes.
    std::size_t predictedBytes = 0;
    /// Bytes charged to the budget (0 when the reservation failed, which
    /// fails the Create and leaves no record).
    std::size_t reservedBytes = 0;
    std::size_t budgetBytes = 0; ///< The budget's capacity at Create.
    std::size_t remainingAtDecision = 0; ///< Remaining() read at the decision.
    /// The batch cap the decision ran at (the scratch clamp's result, or the
    /// caller's own `maxBatchBytes` when the clamp did not bind) — the cap
    /// every Create-time build below ran at, so the charged arena is the
    /// realized one.
    std::size_t maxBatchBytes = 0;
    /// The fired estimate's decomposition, in bytes, so a reader adds them up
    /// and lands on `predictedBytes` instead of taking the total on trust.
    /// The blocked rung's slice and accumulation temporary are the only terms
    /// not named: they are `predictedBytes` minus the sum.
    std::size_t structuralBytes = 0; ///< Pair stores, screened task list, metric/eigen cluster.
    std::size_t rootBytes = 0; ///< The retained metric-inverse root (nAux x nAux).
    /// ONE n^2 x nAux array (the raw tensor on the fast rung; on the blocked
    /// one this is the retained transform, and the fast rung's second copy is
    /// the difference between the rungs).
    std::size_t tensorBytes = 0;
    std::size_t occTransformBytes = 0; ///< BuildFock's n x nOcc x nAux half transform.
    std::size_t fockBytes = 0; ///< The rank-2 working class.
    std::size_t arenaBytes = 0; ///< The 3c batch arena at maxBatchBytes.
    std::size_t sliceFunctions = 0; ///< kBlocked: the shell range's auxiliary function width.
    std::size_t sliceCount = 0; ///< kBlocked: the number of auxiliary shell ranges built.
};

/// One canonicalization/straddle record of the symmetry block.
/// \ingroup qcx-io
struct RunDegenerateSubspace {
    std::string irrepLabel; ///< The full-group irrep label ("E", "T2g", "Pi_u").
    std::vector<int> moIndices; ///< MO columns of the subspace (coefficients order).
};

/// The symmetry block of a run result: the full-group labeling of
/// the converged state, when the stage ran. Carries the per-MO irrep
/// labels, the detected full group and the Abelian reduction the SCF
/// actually used, the canonicalization/straddle record of the degenerate
/// subspaces, and the finite symmetrization subset (linear molecules
/// only). The MO coefficients and the symmetrized density are NOT
/// serialized: the Molden export carries the coefficients, and the JSON
/// consumer needs the labels, not the matrices.
/// \ingroup qcx-io
struct RunSymmetry {
    std::string fullGroup; ///< The detected full group ("D2h", "Cinfv", ...).
    std::string abelianReduction; ///< The computational (Abelian) group the SCF used.
    std::vector<std::string> labels; ///< Per-MO Mulliken labels, one per coefficient column.
    std::vector<int> irrepIndices; ///< Per-MO row into the full-group character
                                   ///< table; -1 on the linear path.
    std::vector<RunDegenerateSubspace> canonicalized; ///< Subspaces whose degenerate
                                                      ///< partners were rotated.
    std::vector<RunDegenerateSubspace> straddled; ///< Aufbau-straddling subspaces:
                                                  ///< recorded, NOT rotated.
    std::vector<std::string> symmetrizationSubset; ///< The finite symmetrization
                                                   ///< subset (linear path only).
    int averagedElementCount = 0; ///< |G| on the table path, the subset size on the
                                  ///< linear path.
};

/// One spin channel's symmetry-blocking DISCLOSURE (schema 32): what the run
/// did with the point-group blocking it was in a position to use, and the
/// numbers the decision rested on.
///
/// It exists because the blocked diagonalization's equivalence to the plain
/// one holds only while the spin Fock commutes with the group, and that is a
/// property of the SOLUTION rather than of the nuclear framework: an
/// unrestricted solution is free to have a smaller invariance group. Blocking
/// a Fock that has broken it does not accelerate that solution — it projects
/// it out and returns the symmetry-adapted one, converged and
/// variational-looking, at the higher energy (measured on stretched H2/STO-3G:
/// +123.7 mHa). The scf module measures the precondition per spin and per
/// diagonalization and refuses the blocks when it fails; this struct is that
/// refusal reaching the run's own record, because a demotion disclosed only to
/// a caller holding the in-process result would let the JSON read as though
/// the blocks had been used.
///
/// The action vocabulary is the guard's enum (`qcx::scf::SymmetryBlockingAction`),
/// emitted as its enumerator words. `"kNotRequested"` is deliberately NOT one
/// of them: a run that never asked carries no block at all, so its absence is
/// the statement (see RunResult::symmetryBlockingAlpha).
/// \ingroup qcx-io
struct RunSymmetryBlocking {
    /// What the run did with the blocking: `"kUsed"` (every diagonalization
    /// ran on the irrep blocks), `"kDemoted"` (some ran blocked and some ran
    /// the plain solve — the run walked both paths), `"kRefused"` (blocking
    /// was requested and never ran), `"kUnavailable"` (the
    /// linear-dependence removal left a non-square orthogonalizer, so no
    /// blocked solve could be served at all). The two words a reader must
    /// tell apart are `"kUsed"` and `"kDemoted"`; `"kUnavailable"` is not
    /// `"kUsed"` either — it names a run that measured no Fock. One of those
    /// four words, always: a channel that disclosed nothing is an unset
    /// optional, never this member left blank.
    std::string action;
    /// Diagonalizations that ran on the irrep blocks.
    int blockedSolveCount = 0;
    /// Diagonalizations that ran the plain n x n solve (the guard's
    /// refusals, and the finalizer's re-diagonalization).
    int plainSolveCount = 0;
    /// The relative commutator norms
    /// `|| [F_orth, A(g_j)] ||_F / || F_orth ||_F` of the WORST measurement
    /// of the channel — so they always explain
    /// `max_generator_commutator_norm` — one per generator of the
    /// computational group. They are here so a reader can VERIFY the
    /// decision rather than take it. Empty when nothing was ever measured
    /// (`"kUnavailable"`).
    std::vector<double> generatorCommutatorNorms;
    /// The largest relative commutator norm the channel measured; compare it
    /// with `tolerance`. 0 when nothing was measured.
    double maxGeneratorCommutatorNorm = 0.0;
    /// The threshold the norms above were compared against, so the numbers
    /// are readable without the source (the guard's constant is internal to
    /// the scf module).
    double tolerance = 0.0;
};

/// The `ri_tensor_mode` block of resources_resolved (schema 19): the
/// requested-vs-ran pairing for the RI tensor mode. It exists because the
/// mode record alone cannot answer the question the enforcement contract
/// cares about — what did the input ASK FOR, and what actually ran — on the
/// very paths where a request would otherwise leave no trace: a run with no
/// mode_record (the legacy null-budget path, the lean arm), and every
/// family that has no RI disk rung at all (a `disk` request on a direct,
/// qfmm, gpu or lean run). Absent when the input named NEITHER request key:
/// an omitted key is the default request, and a default is the
/// system's judgement rather than a request to record. Schema 22 also emits
/// it for `[diagnostics] force_disk_ri` (which moved
/// the force out of the rung selector), so a forced request is paired with
/// what ran on the paths where mode_record is absent.
/// \ingroup qcx-io
struct RunRiTensorMode {
    /// What the input asked for: "auto" | "disk" | "forced_disk". The first
    /// two are `method.ri_tensor_mode`'s own accepted words, and both are
    /// rung selections; the third is the FORCE request, which no longer
    /// exists as a rung word and is carried by `[diagnostics]
    /// force_disk_ri` — the record keeps the word because it names the
    /// request, not the key that spelled it, and `forced` below names the
    /// key.
    std::string requested;
    /// What actually ran: "disk" when the storage-module disk builder was
    /// wired, "in_memory" on every other path (the engine's own rung — fast
    /// or light — is named by mode_record.mode where a mode record exists).
    std::string resolved;
    /// The outcome word: "honoured" | "ladder_fit" | "not_applicable" |
    /// "demoted".
    /// - "honoured": the request got what it named (auto -> in_memory,
    ///   disk -> disk, forced_disk -> disk).
    /// - "ladder_fit": `disk` was permitted and a memory rung rode — the
    ///   knob's DOCUMENTED inert-by-design contract, so it is not a
    ///   demotion: the request did not have to change, a fitting memory
    ///   rung is what the request itself prescribes.
    /// - "not_applicable": the resolved family has no RI disk rung, so the
    ///   request had nothing to select (recorded, not silent — this is the
    ///   row the enforcement audit found missing).
    /// - "demoted": the request could not be honoured on this run and the
    ///   best runnable arrangement ran instead (demotion is the
    ///   expected outcome, with the record showing BOTH sides).
    std::string outcome;
    /// Why, one line; absent when the outcome is "honoured" (the null
    /// honesty policy — an absent reason is never a fabricated "fine").
    std::optional<std::string> reason;
    /// Whether the request came from `[diagnostics] force_disk_ri` rather
    /// than from `method.ri_tensor_mode` (schema 22). ALWAYS present in the
    /// block, on the `mode_record.forced_disk` rule: a request's source is
    /// not inferable from `requested` alone, because the word "forced_disk"
    /// now appears in exactly one place — this block — and a reader must be
    /// able to tell the key that carried the request without reading the
    /// input file. A forced request that demoted still carries `true`: the
    /// member names the KEY, never the outcome (which `outcome` states).
    bool forced = false;
};

/// The `ri_chunk_bytes` request record of resources_resolved (schema 23): the
/// outcome of `method.ri_chunk_bytes`, the disk rung's chunk-size hint. It is a
/// sibling of RunRiTensorMode above, and it exists for the same reason: the
/// hint is read at exactly one site — the ri_j_link family's disk-rung
/// construction (`run_driver.cpp` engageDiskRung, the `DiskRiFockOptions`
/// assignment) — so on every other family, and on ri_j_link when a fitting
/// in-memory rung rides, the key used to leave NO trace anywhere in the
/// document. The run proceeded and the key was dropped in silence.
///
/// The drop itself is correct: a pure
/// size hint on a family with no chunking to size is not refused, because
/// refusing it is pedantic. What the contract requires is the DISCLOSURE, and
/// this block is it — the key stays non-fatal and the record states which of
/// the two things happened.
///
/// The block carries no `resolved` member, unlike its sibling: there is no
/// second VALUE to pair with `requestedBytes` (a rung word is what the sibling
/// pairs, and a chunk size has no such counterpart). The discriminating fact is
/// the outcome word alone, and the reason names the store on the drop side.
/// Absent when the input did not name the key: an omitted key is the default,
/// not a dropped request (the schema-19 null-honesty rule).
/// \ingroup qcx-io
struct RunRiChunkBytes {
    /// What the input asked for, in bytes — `method.ri_chunk_bytes` exactly as
    /// parsed and validated (the Create() contract is >= 1, so a serialized
    /// block never carries 0). Recorded in the dropped case too: a disclosure
    /// that did not say WHAT was dropped would not let a reader tell this key
    /// apart from any other.
    std::size_t requestedBytes = 0;
    /// The outcome word: "honoured" | "dropped".
    /// - "honoured": the storage-module disk-backed RI-J store was constructed
    ///   and received `requestedBytes` as its chunk size — the disk rung
    ///   engaged, which is the one code path that reads the key.
    /// - "dropped": the run proceeded without the hint. `reason` says which of
    ///   the two drop states it was.
    ///
    /// Deliberately NOT the sibling's four words: on the ri_tensor_mode
    /// pairing `ladder_fit` means "the request was NOT lost" (a fitting memory
    /// rung is what a `disk` request prescribes), while the same situation here
    /// is a genuine drop. One word with the opposite polarity in two sibling
    /// blocks would be a trap for exactly the reader this field exists for.
    std::string outcome;
    /// Why, one line; absent when the outcome is "honoured" (the null honesty
    /// policy — an absent reason is never a fabricated "fine").
    std::optional<std::string> reason;
};

/// The `ri_orbit_expansion` block of resources_resolved (schema 26): the
/// requested-vs-ran pairing for the RI-J orbit expansion
/// (`method.ri_orbit_expansion`). It exists for the
/// same reason the ri_tensor_mode and ri_chunk_bytes blocks do — the key's
/// consumption is one branch deep (the ri_j_link arm) and the engine it drives
/// has an ON-by-default permission with no reduction of its own, so without
/// this block a requested-and-inert run and a requested-and-engaged run would
/// serialize identically. Absent when the input did not name the key: an
/// omitted key is the default request, and a default is the system's
/// judgement rather than a request to record (the schema-19 null-honesty rule).
/// \ingroup qcx-io
struct RunRiOrbitExpansion {
    /// What the input asked for — `method.ri_orbit_expansion` exactly as
    /// parsed. Recorded alongside the outcome rather than replaced by it: a
    /// false request and a run that never wrote the key are different facts
    /// about the input, and only the request's presence separates them.
    bool requested = false;
    /// The outcome word: "engaged" | "not_requested" | "inert_trivial_group".
    /// - "engaged": the driver built both reductions, handed them to the RI-J
    ///   builder, and the engine reports the expansion ran
    ///   (RiJkFockBuilder::OrbitExpansionEngaged) — the 3c task grid was
    ///   walked to one representative per joint orbit.
    /// - "not_requested": the key was absent or false, so the plain walk ran
    ///   (every cell evaluated). The default, and not a failure.
    /// - "inert_trivial_group": the key was true and both reductions were
    ///   built, but a group order of 1 makes the mechanism the identity —
    ///   every cell is its own orbit — so nothing engaged. Recorded because
    ///   the request was made and did not do what it named; `reason` says so.
    std::string outcome;
    /// Why, one line; absent when the outcome is "engaged" (the null honesty
    /// policy — an absent reason is never a fabricated "fine").
    std::optional<std::string> reason;
};

/// One row of RunEriStore::hitQuartetsByClass below: the ERI block class's
/// two angular momenta and the quartets the store served for it. A row per
/// class rather than an object keyed by a composite string, so the class a
/// count belongs to is readable without a key-encoding convention the
/// document would otherwise have to state and every consumer re-derive.
/// \ingroup qcx-io
struct RunEriStoreClassHits {
    int lBra = 0; ///< The bra side's angular momentum (CachedEriStats' pair.first).
    int lKet = 0; ///< The ket side's angular momentum (CachedEriStats' pair.second).
    std::size_t hitQuartets = 0; ///< Quartets the store served for this class.
};

/// The `eri_store` block of resources_resolved (schema 33): the disk-tier ERI
/// store request's requested-vs-ran pairing (`method.eri_cache_store`, the
/// engine-decorator seam's request surface). It exists because the
/// key's consumption site is a decorator the ENGINE cannot see: the driver
/// constructs `storage`'s CachedEriBatchEngine, hands `integrals` an opaque
/// engine pair (the module boundary — `integrals` cannot name `storage`), and
/// owns the decorator for the run, so without this block a run that asked for
/// a store and a run that never asked would serialize identically.
///
/// The disclosure contract is the block's whole point: an enforced request is
/// USED (`engaged` reads "disk"), or DEMOTED to the closest workable
/// arrangement with the demotion DISCLOSED (`demoted` plus `demotedReason`,
/// beside the path that was requested). A silent substitution — running the
/// in-memory tier under a disk request with nothing in the record to say so —
/// is the defect this block exists to make impossible. A refusal BY NAME is a
/// run that never reached serialization, so it has no shape here.
///
/// Absent when the input named no store (an absent or empty
/// `method.eri_cache_store`): an omitted key is not a request, and the
/// a default the system's judgement rather than a request to record (the
/// schema-19 null-honesty rule every request block in this section follows).
/// \ingroup qcx-io
struct RunEriStore {
    /// The store path the input named, verbatim — recorded in the demoted
    /// case too: a disclosure that did not say WHAT was requested would not
    /// let a reader tell this key apart from any other.
    std::string path;
    /// What actually RAN: "disk" | "in_memory_cache" | "none".
    /// - "disk": the storage-module decorator served the run's ERI batches —
    ///   the request was honoured, and `demoted` is false.
    /// - "in_memory_cache": the run proceeded on a tier that is not the
    ///   store, so it was demoted to the closest workable arrangement.
    /// - "none": no cache tier served the run at all — a demotion whose
    ///   closest workable arrangement is the plain engine.
    ///
    /// The last two are the demotion's own disclosure and always carry
    /// `demoted` = true; reading `engaged` alone therefore cannot mistake a
    /// demoted run for an honoured one.
    std::string engaged;
    /// True when the requested store was NOT what ran. False on an honoured
    /// request. Written on every serialized block, so a consumer keyed on it
    /// never has to infer the demotion from the `engaged` word's value.
    bool demoted = false;
    /// The demotion's cause, verbatim from the store's own error (a refused
    /// open, a write failure, a fingerprint mismatch). Absent on an honoured
    /// request (the null honesty policy — an absent reason is never a
    /// fabricated "fine").
    std::optional<std::string> demotedReason;
    /// Quartets served from the store's dataset over the whole run, mirrored
    /// from the decorator's own CachedEriStats after the SCF. 0 when the
    /// store did not engage — never a fabricated reading, and never a
    /// substitute for `engaged`.
    std::size_t hitQuartets = 0;
    /// Quartets that missed the store and were recomputed through the
    /// underlying engine, the decorator's own count.
    std::size_t missQuartets = 0;
    /// Wall time the store spent serving hits, checksums included
    /// (CachedEriStats::readMs). Wall time, so it is comparable between runs
    /// only at one thread count.
    double readMs = 0.0;
    /// Wall time spent in the underlying engine on the misses
    /// (CachedEriStats::recomputeMs).
    double recomputeMs = 0.0;
    /// The hits broken down by ERI block class, one row per (lBra, lKet) the
    /// store served, in the store's own ordering. Empty when there is no
    /// per-class row to report (the store did not engage, or engaged and
    /// recorded none) — an empty array is OMITTED rather than written as
    /// one, the block's own absence rule applied one level down.
    std::vector<RunEriStoreClassHits> hitQuartetsByClass;
};

/// input the certified fp32 lane's default was RESOLVED against, recorded so
/// a reader can check the verdict against the machine that produced it. It is
/// the lane's *decided* record; RunCertifiedBound is the same lane's
/// *delivered* record, and the two together are what make a verdict auditable.
///
/// The quantity is the measured fp32/fp64 FMA-throughput ratio, never an
/// "AVX" or "GPU" label: a label has already been wrong in this tree, in
/// exactly this place (the FMA flag read from the wrong CPUID leaf, which
/// made the whole AVX2 micro-GEMM tier dead code on a machine that has FMA).
/// A measured ratio cannot be wrong that way; it can only be noisy, which is
/// why the pair count and the spread ride along.
///
/// `measured` is what keeps a MEASURED 1.0 apart from the UNMEASURED 1.0: a
/// scalar machine really does measure ~1.0, and a probe that never ran
/// reports the same 1.0 (the documented unknown). The verdict is off either
/// way; the record is not, and a reader that could not tell them apart would
/// be reading an assumption as a measurement.
/// \ingroup qcx-io
struct RunComputeProfile {
    /// Which probe produced the numbers: "host" (the CPU-side register-bound
    /// SIMD FMA micro-benchmark, backend/host_compute_profile.hpp) or
    /// "device" (the CUDA device probe) - the driver picks it from the family
    /// the run wires, so a device number can never arrive as a host reading
    /// or the reverse. Empty means never filled: the no-probe case (a run
    /// that resolved no probe) is the block's ABSENCE, which is the same
    /// null-honesty reading every other optional block in this schema uses,
    /// and there is deliberately no third value - `Create` refuses one. The
    /// "none" default this member used to carry was reachable only through a
    /// producer that filled the readings and forgot the arm, which is
    /// precisely the producer that could not be told apart from an honest
    /// one by the emitted record.
    std::string source;
    /// The measured fp32 FMA throughput in GFLOP/s (0.0 = unmeasured).
    /// Scalar FLOP, lane width included - the device probe's own counting
    /// convention. Never compare across machines: it is clock- and
    /// core-dependent, while the ratio below is not.
    double fp32Gflops = 0.0;
    /// The measured fp64 FMA throughput in GFLOP/s (0.0 = unmeasured).
    double fp64Gflops = 0.0;
    /// The measured fp32/fp64 throughput ratio (1.0 = unmeasured).
    double ratio = 1.0;
    /// True when an actual measurement produced the numbers above.
    bool measured = false;
    /// True when the host probe ran its SIMD lane (256-bit AVX2 FMA) rather
    /// than its scalar reference lane. Always false for a device source.
    bool simdLane = false;
    /// The timed lane pairs the ratio was estimated from (0 = unmeasured).
    int pairs = 0;
    /// The spread of the per-pair ratios (max - min) - the instrument's own
    /// noise, so a reading can be judged rather than trusted.
    double ratioSpread = 0.0;
    /// The verdict the ratio resolved to: the certified fp32 lane's DEFAULT
    /// for this machine. A caller's explicit request is not recorded here -
    /// it resolves at one point and is not a property of the machine.
    /// `certifiedLaneForced` below is what keeps that sentence from being a
    /// reader trap: it is the one flag in this block that is NOT a property
    /// of the machine, and it says when the lane's state did not come from
    /// this verdict at all.
    bool certifiedLaneDefault = false;
    /// Schema 21: true when the input FORCED
    /// the lane through `[method] force_certified_lane` rather than letting
    /// the probe's verdict above decide it. The distinction is the whole
    /// reason the key exists - both arms of the certified fp32 lane had only
    /// ever been exercised by injecting the probe's return value in source,
    /// so on a machine whose probe resolves OFF this block could only ever
    /// report the OFF verdict.
    ///
    /// Always present on a serialized block, and that is deliberate, on the
    /// `mode_record.forced_disk` rule: an absent member could not
    /// distinguish a forced run from a probed one. A forced block reads
    /// `certified_lane_forced: true` BESIDE the machine's own
    /// `certified_lane_default` - the block keeps reporting what the probe
    /// measured and what that verdict would have been, and this flag says
    /// which of the two the lane's state came from. Never infer the lane's
    /// state from `certified_lane_default` alone.
    bool certifiedLaneForced = false;
    /// The threshold the ratio was compared against - qcx-integrals' one
    /// kCertifiedLaneMinRatio (4.0), carried here because io cannot include
    /// the integrals policy, so a consumer does not have to re-derive the
    /// rule to read the block. 0.0 = unset, and `Create` refuses it: it is
    /// the value every ratio passes, so an emitted 0.0 would read as the
    /// certified lane ENABLED on hardware whose ratio the lane cannot pay
    /// for - the one number in this block whose wrong value flips a verdict
    /// rather than merely misdescribing a reading.
    double certifiedLaneMinRatio = 0.0;

    /// Validates a filled block and returns it, or the first invariant the
    /// candidate breaks - the repo's fallible-construction form, which
    /// replaces exceptions everywhere.
    ///
    /// The checks are the field contracts above, which the schema cannot
    /// enforce and which held only because this block had ONE producer:
    /// `source` names a resolved ARM and never an unfilled default, and
    /// `certifiedLaneMinRatio` is the positive threshold the verdict was
    /// read against and never its "unset" 0.0. The device arm's three
    /// not-reported readings are checked with them, for the same reason:
    /// both this header and the schema document state that
    /// `simdLane`, `pairs` and `ratioSpread` are the host arm's readings
    /// (the device probe measures once and reports none of them), so a
    /// device block carrying them would report a host SIMD lane width and a
    /// host noise estimate for a GPU.
    ///
    /// The guard sits on the TYPE rather than in the producer's comment,
    /// deliberately: a second producer - a new call site, a refactor, a test
    /// fixture - is exactly the event that breaks these invariants, and it
    /// inherits this check where it would not inherit a comment. Validation
    /// rewrites nothing: a correct producer's block passes through unchanged,
    /// so the schema version does not move. (The block is not deterministic
    /// run to run and never was - its numbers are a measurement; what is
    /// fixed is the key set, the arm, the threshold and the verdict rule.)
    /// \param candidate The filled block.
    /// \returns The block, or kInvalidArgument naming the invariant it broke.
    static qcx::Result<RunComputeProfile> Create(RunComputeProfile candidate) {
        if (candidate.source != "host" && candidate.source != "device")
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "compute_profile.source must be \"host\" or \"device\" - got \"" +
                               candidate.source +
                               "\"; a run that resolved no probe records NO block (the absent "
                               "block), never a present one carrying an unfilled default"});
        }

        if (!(candidate.certifiedLaneMinRatio > 0.0))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "compute_profile.certified_lane_min_ratio must be positive - got " +
                               std::to_string(candidate.certifiedLaneMinRatio) +
                               "; 0.0 is the unset sentinel and every ratio passes it, so an "
                               "emitted 0.0 would read as the certified fp32 lane enabled on "
                               "any hardware"});
        }

        if (candidate.source == "device" &&
            (candidate.simdLane || candidate.pairs != 0 || candidate.ratioSpread != 0.0))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "a compute_profile block with source \"device\" carries no host "
                           "readings: simd_lane, pairs and ratio_spread stay at their "
                           "not-reported values (the device probe measures once)"});
        }

        return candidate;
    }
};

/// The resources_resolved block: what the run actually applied — the
/// input caps plus the enforcement record. Always present: even the
/// defaults are resolved and recorded, so a consumer can tell a capped
/// run from an uncapped one.
/// \ingroup qcx-io
struct RunResourcesResolved {
    double memoryCapGiB = 16.0; ///< The memory cap in effect (the input value).
    int threadCap = 0; ///< The thread ceiling in effect (the input value).
    /// True when the driver's own job-object process-memory cap was
    /// applied in-process (Windows). False when the cap could not be
    /// applied — the platform has no job objects (Linux CI), the process
    /// was already in an external job (the harness gate or the CI
    /// runner), or a job-object call failed — and capNote names the
    /// reason. The benchmark harness gate is independent by design
    /// (defense in depth, same constant value).
    bool inProcessCapApplied = false;
    /// Why the in-process cap was not applied; absent when it was.
    std::optional<std::string> capNote;
    /// The workspace budget the ri_j branch granted the engine;
    /// present only on the budget path, after the run.
    std::optional<RunWorkspaceBudget> workspaceBudget;
    /// The engine's Create-time mode decision and its firing estimate
    /// terms; present only on the budget path, after the run.
    std::optional<RunModeRecord> modeRecord;
    /// The EXCHANGE builder's Create-time mode record of a UHF run: the
    /// UHF contraction builds its Fock from TWO builders (the coulomb
    /// half and the exchange half, each admission-gated against the same
    /// budget), so the coulomb half's record (modeRecord) alone shows
    /// only half the admission evidence - an UHF run's
    /// classTableBytes/classPathDisengaged for the exchange half lives
    /// here. The composed-QFMM UHF run emits its exchange-only
    /// half's record here the same way. Absent for the single-builder RHF
    /// runs (direct / ri_j_link: one Create-time decision), on the
    /// composed-QFMM RHF run (its exchange half rides the shared budget
    /// but is not surfaced separately), and off the budget path.
    std::optional<RunModeRecord> exchangeModeRecord;
    /// The composed full-RI builder's own Create-time rung record (schema
    /// 30): the ri_jk family's decision, mirrored from the integrals module's
    /// RiFullFockModeInfo, and the sibling of modeRecord above for a different
    /// builder with its own rung vocabulary (see RunRiJkMode). Absent on every
    /// family that wired no composed full-RI builder, and on a ri_jk run whose
    /// builder was created without a workspace budget (the legacy null-budget
    /// path: no decision was made).
    ///
    /// Deliberately NOT folded into modeRecord: the two records describe two
    /// different builders whose rung words and term decompositions do not
    /// correspond, so one block would make a reader derive each number's
    /// meaning from `selection.builder`.
    std::optional<RunRiJkMode> riJkMode;
    /// The builder-resolution record (the "heuristic auto-selection"
    /// name is RETIRED): what the driver picked, the within-family member it
    /// wired, and why, or the explicit override with its divergence warning.
    /// Under the lean flip nothing ranks candidates any more - the
    /// record carries the resolution that was made, not a model's decision,
    /// which is why `picked` is the family default at every size and the
    /// `scores` array has no producer. Absent when the builder was not
    /// resolved (a run refused before the wiring).
    std::optional<RunSelection> selection;
    /// The RI tensor-mode request record (schema 19): the requested-vs-ran
    /// pairing of `method.ri_tensor_mode` (and, schema 22, of
    /// `[diagnostics] force_disk_ri`). Present whenever the input NAMED
    /// either request key, whatever the family and whatever the budget path;
    /// absent when both were omitted (an omitted key is the default request,
    /// and a default is the system's judgement, not a request to
    /// record).
    /// Deliberately NOT inside modeRecord: that record is absent on exactly
    /// the paths where a request would otherwise vanish (the legacy
    /// null-budget path and the lean arm), which is the gap this block
    /// closes.
    std::optional<RunRiTensorMode> riTensorMode;
    /// The `ri_chunk_bytes` request record (schema 23): the outcome of the
    /// disk rung's chunk-size hint. Present whenever the input NAMED
    /// `method.ri_chunk_bytes`, whatever the family and whatever the budget
    /// path; absent when the key was omitted. Sibling of riTensorMode above,
    /// deliberately a separate block rather than a new member on it: the two
    /// report on two different keys, an input may name either, both or
    /// neither, and folding the size hint into the rung request's pairing
    /// would make one block's absence mean two different things.
    /// Deliberately NOT `resources_resolved.selection.warning`: that member
    /// has a single producer (the device-less gpu fallback), so a second
    /// writer would clobber one of the two texts on a run that produced both.
    std::optional<RunRiChunkBytes> riChunkBytes;
    /// The `ri_orbit_expansion` request record (schema 26): what the input
    /// asked of the RI-J 3c orbit expansion and what ran. Present whenever
    /// the input NAMED `method.ri_orbit_expansion`, whatever the family and
    /// whatever the budget path; absent when the key was omitted. Sibling of
    /// riTensorMode and riChunkBytes above, for the same reason: a third key,
    /// a third block, so absence never means two things. Written by the
    /// ri_j_link arm from the engine's own disclosure
    /// (RiJkFockBuilder::OrbitExpansionEngaged) rather than recomputed in the
    /// driver, so the record cannot disagree with the engine about whether the
    /// reduction ran.
    std::optional<RunRiOrbitExpansion> riOrbitExpansion;
    /// The disk-tier ERI store request record (schema 33): what the input
    /// asked of the engine-decorator seam and what actually ran. Present
    /// whenever the input NAMED a store (`method.eri_cache_store` non-empty),
    /// on BOTH the honoured and the demoted arm — the block is the
    /// demotion's disclosure surface, not only the success path's evidence —
    /// and absent when the key was omitted or empty. Sibling of riTensorMode,
    /// riChunkBytes and riOrbitExpansion above, for the same reason: a fourth
    /// key, a fourth block, so absence never means two things. Unlike its
    /// siblings the counters are not a driver recomputation — the driver
    /// owns the decorator and mirrors the store's own Stats() after the SCF,
    /// so the record cannot disagree with the store about what it served.
    std::optional<RunEriStore> eriStore;
    /// The compute profile the certified lane's default was resolved against
    /// (schema 20): present whenever a probe was resolved - the
    /// host arm on a CPU-family run, the device arm behind its cap gate on a
    /// GPU-family run - and absent when none applied (the policies still
    /// resolve; the driver simply had no probe to seed them with, which is
    /// the conservative interim rather than a missing measurement).
    std::optional<RunComputeProfile> computeProfile;
};

/// The term_counters block of an instrumented ri_j run: the
/// integrals engine's RiTermCounters (ri_engine.hpp) mirrored into the
/// schema layer — io stays pure, the RunModeRecord pattern. Exactly the
/// five engine-emitted ids x/p3/g3/qx/gx of the cost-table backbone's ri_j
/// vector; each is the run's non-negative INTEGER per-run total over the
/// whole SCF run (not per iteration), 0 when the term's work never
/// occurred — never omission, never null. The s/p geometry terms and the
/// qx_occ occupancy product are model terms, never engine-emitted. Present
/// exactly when the run was instrumented: an ri_j run whose trace file was
/// set (every SCF BuildFock call then carries the per-call stats sink the
/// exchange counts ride).
/// \ingroup qcx-io
struct RunTermCounters {
    std::size_t x = 0; ///< The aux pair-class engagements (the x term).
    std::size_t p3 = 0; ///< The screened 3c orbital pair-class engagements (the p3 term).
    std::size_t g3 = 0; ///< The 3c ERI kernel weight (the g3 term).
    std::size_t qx = 0; ///< The exchange's screened quartet count (the qx term).
    std::size_t gx = 0; ///< The exchange's ERI kernel weight (the gx term).
};

/// The certified mixed-precision bound block of a run (schema 14:
/// the delivered bound; schema 16: the global budget enforcement's own
/// reading of the same calls): the fp32 lane's accumulated
/// density-weighted kernel-bound sum, mirrored from the direct builder's
/// `certifiedBoundSumOut` out-parameter. The quantity is what the driver
/// previously computed and discarded; it is a per-Fock-build upper bound
/// on the Fock-element error the fp32 lane delivered, in hartree. Each
/// member is a run-level reduction over the calls the seam observed — the
/// block carries no per-quartet detail.
///
/// The block carries TWO quantities that are not interchangeable, and a
/// reader must keep them apart: `lastCallHa`/`maxCallHa` are the DELIVERED
/// kernel-bound sums (what the lane's arithmetic cost the Fock matrix),
/// while `routedHa` is the ROUTING bound sum the preset-derived budget was
/// compared against (the gate's own units, orders smaller per routed
/// quartet). `enforced` says whether the comparison ran at all.
/// \ingroup qcx-io
struct RunCertifiedBound {
    /// The number of Fock builds the block summarizes: every main-SCF
    /// BuildFock call that carried the bound out-parameter, converged or
    /// last-iterate alike. Never 0 in a serialized block (the block is
    /// absent when the seam observed none).
    long long calls = 0;
    /// The final observed call's bound sum — the build whose Fock matrix
    /// the reported energy belongs to. This is the per-call quantity the
    /// Preset sweep records (the "bound sum" column).
    double lastCallHa = 0.0;
    /// The largest bound sum over the observed calls — the conservative
    /// single-call reading of the lane's delivered error.
    double maxCallHa = 0.0;
    /// The global budget enforcement's own members (schema 16), all
    /// measured on the SAME final observed call as lastCallHa. See
    /// RunCertifiedBound::enforced.
    bool enforced = false;
    /// The budget the accuracy preset derived for the final observed call
    /// (Eh): PresetEnergyBudget(accuracy) less the call's screened-out
    /// bound sum less the ladder's slack. 0.0 is a real budget here (the
    /// screening co-term consumed the target) — it is never a
    /// not-measured marker.
    double budgetHa = 0.0;
    /// The final observed call's ROUTED bound sum (Eh) — the quantity the
    /// budget was compared against, in the routing gate's own bound units.
    /// Not the same quantity as lastCallHa: that one is the delivered
    /// kernel-bound sum, roughly four to five orders larger per quartet on
    /// the fixtures measured (certified_budget_test.cpp prints both).
    double routedHa = 0.0;
    /// The final observed call's routed quartet count — the size of the
    /// fp32 task list the routing pass produced, i.e. what the comparison
    /// was run ON, not what survived it: the count stays positive on a
    /// fall-back (the lane was admitted and then refused), and reads 0
    /// only when the gate admitted nothing. `fell_back_to_fp64` and a zero
    /// `last_call_ha` are what say the lane then delivered nothing.
    long long routedQuartets = 0;
    /// The comparison's verdict on the final observed call: true when the
    /// routed sum did not fit the budget, so the build ran its whole
    /// quartet set on the fp64 lane — the same set on the same code path
    /// as a lane-disabled build. False keeps the ordinary certified
    /// routing.
    bool fellBackToFp64 = false;
};

/// The QFMM model block (schema 29): what the composed-QFMM builder's
/// Coulomb half ACTUALLY ran, mirrored from
/// `QfmmHfFockBuilder::ModelRecord()` - the engine's own record, forwarded
/// from the nested `QfmmJBuilder` and never recomputed here (the
/// `ri_orbit_expansion` rule, which exists so a record cannot disagree with
/// the engine about what ran).
///
/// It exists because the builder's DEFAULTS changed - the multipole
/// geometry model from the recorded
/// midpoint-padded extent to the product-distribution ball, and the
/// separation test from the recorded centre-to-width form to the
/// surface-to-surface ball test - so the form a run used stopped being a
/// fact about any input key and became a fact about the build alone. The
/// two models are not interchangeable: they put different objects under the
/// multipole expansion (the shell-pair midpoint with a padded conservative
/// bound, versus the pair's own significant product distribution) and admit
/// different far fields, so a reader comparing energies or far-field counts
/// across runs must be able to tell which one produced them.
///
/// The member values keep the engine's ENUMERATOR vocabulary - the
/// `mode_record.mode` precedent (`kFastPath`/`kLightPath`, not a new
/// lowercase vocabulary), so the word a record carries can be grepped in
/// the source that chose it.
/// \ingroup qcx-io
struct RunQfmmModel {
    /// The geometry model the octree and the interaction lists were built
    /// over: "kProductBall" (the centre and radius of the pair's significant
    /// product distribution - the object the multipole expansion is actually
    /// of, and the default) or "kMidpointBound" (the
    /// retired conservative midpoint-padded bound, still selectable).
    std::string extentModel;
    /// The separation test the well-separatedness classification ran:
    /// "kSurfaceBall" (the surface-to-surface test: distance against the
    /// nodes' charge radii plus a buffer - the default) or
    /// "kWidthTheta" (the retired centre-to-width test, still selectable, and
    /// what an explicit `[method] theta` > 0 selects).
    std::string separationMode;
    /// The kSurfaceBall buffer, in units of max(rA, rB), as the build
    /// resolved it: 0 is the bare touching test, positive is the multipole
    /// error bound's buffer, and a NEGATIVE value is the degenerate gate
    /// (nothing well separated - the far field is empty and the build is the
    /// exact restricted near-field direct build). Both the explicit theta < 0
    /// form and every absent-theta run at a preset whose own rung is the gate
    /// resolve to that negative form.
    ///
    /// Read it only under "kSurfaceBall": the engine records the parameter of
    /// the test that RAN, so this member is 0.0 under "kWidthTheta" - the
    /// don't-care of a test that never read it, NOT a claim that the touching
    /// test was used.
    double separationK = 0.0;
    /// The resolved theta of the built octree - the centre-to-width test's
    /// own parameter: the explicit value an input wrote, which is the ONLY way
    /// "kWidthTheta" is ever selected, so this member is never a preset
    /// default while that word is in the record.
    ///
    /// The same one-sided rule as `separationK`: it reads 0.0 under
    /// "kSurfaceBall", where the buffer above is what separated the tree. Read
    /// the two members together with `separation_mode`, never one alone - a
    /// negative theta is the gate under EITHER word, and that run's record
    /// carries "kSurfaceBall" with the negative buffer.
    double theta = 0.0;
};

/// The `xc_grid` block of a run result (schema 34): the XC integration
/// grid the run BUILT, in the engine's own six settings.
///
/// It exists because the grid was the one physics-bearing input a run could
/// neither choose nor disclose: the keys were compile-time defaults with no
/// input path at all, so a record's energy could not say what quadrature it
/// was integrated on (the equivalence the whole XC path rests on - the same
/// input at two Lebedev sizes is two different numbers), and two runs whose
/// only difference was the grid were indistinguishable in the artifact a user
/// keeps.
///
/// These are the RESOLVED settings - the values the engine was created with,
/// never the file's request re-read: the parser resolves absent keys to the
/// engine's own defaults once (`RunGridInput`), the driver carries that struct
/// into `XcGridSettings` and into this block, so a key the file omitted reads
/// back as the default that acted and never as a fabricated difference.
/// A run that built no XC grid - every Hartree-Fock run, whose method has no
/// density functional to integrate - carries NO block rather than a block of
/// defaults: the absence is the statement that no grid was involved, and the
/// `method` member beside it is what tells a consumer which of the two it is
/// reading.
/// Public aggregate: the fields are the API.
/// \ingroup qcx-io
struct RunXcGrid {
    std::size_t radialPoints = 75; ///< Radial points per atom.
    std::size_t angularPoints = 302; ///< The Lebedev size the grid was built at.
    double alpha = 0.5; ///< The MHL radial mapping scale, Bohr.
    std::size_t radialExponent = 2; ///< The MHL mapping exponent m.
    double trimWeight = 1e-15; ///< Points with |weight| below this were dropped.
    std::size_t blockTarget = 1024; ///< The spatial re-batching target.
};

/// The run's RESOLVED builder selection in the orthogonal axis vocabulary
/// (schema 35), serialized as "builder_axes":
/// which axes the run actually ran, stated in the vocabulary that replaced the
/// conflated `[method] fock_builder` key.
///
/// **Why the record carries it.** The selection record names the builder in the
/// LEGACY word - `selection.builder` is `ToString(BuilderKind)`, where a lean run
/// reports "direct" - so a reader of a record cannot see the
/// tier at all without already knowing which words were tier words. This block
/// states the same selection one axis at a time, so a record says what RAN rather
/// than what the input spelled.
///
/// **The tier member reports the RESOLVED selection's tier** and nothing else.
/// The two tiers a rung key reaches are deliberately NOT restated, because one
/// fact has one home: the disk tier is `resources_resolved.ri_tensor_mode`'s
/// answer (its ladder decides at the engine's Create-time estimate, after this
/// selection is made) and the blocked tier is the full-RI family's own rung. A
/// reader asking "did this run land on disk?" reads the rung block; a reader
/// asking "what did the selection resolve?" reads this one.
/// \ingroup qcx-io
struct RunBuilderAxes {
    /// The family axis: "direct" | "ri_j_link" | "ri_jk" | "qfmm".
    std::string integralFamily;
    /// The RESOLVED selection's tier - not the tier a key asked for: "lean" |
    /// "in_memory" (see the type's note for where the other two are reported).
    std::string storageTier;
    /// The backend axis: "cpu" | "gpu". There is no "gpu_split" here: the axis
    /// names that candidate and the run path refuses it, so no record carries it.
    std::string executionBackend;
    /// The DEPRECATED key's own word, when the input used it: the exact spelling
    /// the file wrote at `[method] fock_builder` ("direct", "ri_j_link", "ri_jk",
    /// "qfmm", "gpu", "lean", or the tier alias "in_memory"). Present so a reader
    /// can see which vocabulary the input was written in. Absent when the input
    /// used the `[builder]` axes, and absent when it named no builder key at all
    /// - which is what `requestedBy` below is for.
    std::optional<std::string> legacySpelling;
    /// Where the selection came from, one word: "axes" (the `[builder]` axis
    /// keys) | "fock_builder" (the deprecated conflated key) | "ladder" (no
    /// builder key at all - the size ladder decided, selection_resolution.hpp).
    /// Stated rather than left to be inferred from `legacySpelling`'s absence,
    /// which cannot tell the axes from the ladder.
    std::string requestedBy;
};

/// The machine-readable result of one run.
/// \ingroup qcx-io
struct RunResult {
    /// The JSON schema version, serialized as "schema_version". Bump on
    /// any change a JSON consumer must notice (a key added or removed, a
    /// unit or a nullability rule changed) and update the schema document
    /// in the same change. Schema 31 added
    /// `resources_resolved.selection.approximation.exchange_error` (five keys,
    /// the approximated-exchange path's measured per-atom error and the bar it
    /// is read against) - keys a consumer must notice, so the number moved.
    /// Schema 32 added `symmetry_blocking` with its `alpha`/`beta` channels
    /// (RunSymmetryBlocking): the guard's decision, emitted as the keys
    /// a consumer must read to tell a run whose blocking was DEMOTED from one
    /// whose blocking was used. Keys added, so the number moved with them.
    /// Schema 33 added `resources_resolved.eri_store` (RunEriStore): the
    /// disk-tier ERI store request's requested-vs-ran pairing — the path
    /// asked for; the `engaged` word naming what actually RAN; `demoted`
    /// with the store's own verbatim error; and the decorator's
    /// hit/miss/timing counters with their per-class breakdown. Keys added,
    /// so the number moved with them.
    /// Schema 34 added the run's own PHYSICS, under ONE bump for two
    /// payloads: the top-level `method` word (RHF/UHF/RKS/UKS -
    /// the input vocabulary, `ToString(MethodType)`), the top-level
    /// `functional` on a Kohn-Sham run, and the `xc_grid` block
    /// (RunXcGrid) with the six grid settings the run BUILT. Keys added, so
    /// the number moved with them — and the two payloads ride one bump
    /// because they close the same defect: a record that named its Fock
    /// builder but not its physics, and a grid a run could neither choose nor
    /// disclose. `functional` and `xc_grid` are ABSENT on a Hartree-Fock run
    /// (no density functional was named, no grid was built) rather than null
    /// or zero-filled — the null-honesty rule the `symmetry_blocking` absence
    /// uses, and the reason this is a presence rule a consumer must notice.
    ///
    /// Schema 35 adds the top-level `builder_axes` block (RunBuilderAxes): the
    /// resolved builder selection in the orthogonal axis vocabulary, with the
    /// deprecated `[method] fock_builder` word recorded as deprecated. It is a
    /// key a consumer must notice — the whole point of the block is that the
    /// tier and the backend are readable off the record instead of having to be
    /// inferred from which words `selection.builder` happened to use.
    ///
    /// Schema 36 REMOVES `resources_resolved.memory_model` (RunMemoryModel)
    /// and every key under it: the driver's hand-derived prediction of the
    /// run's commit peak, which was deleted rather
    /// than repaired. A removed key is a change a consumer must notice, so
    /// the number moves with it. Nothing reads the block for a physics
    /// result: the memory limit is enforced by the job-object cap the
    /// driver applies (`process_caps`, unchanged), and the rung the run
    /// takes is decided by the integrals engine's own Create-time
    /// estimates. The one fact the block carried that no other key
    /// carries is the within-family member of the direct family — and
    /// `selection.builder_member` (schema 15) is where that member is
    /// named, which is why `lean` needed no replacement.
    static constexpr int kSchemaVersion = 36;

    bool converged = false; ///< True when a convergence criterion fired.
    int iterations = 0; ///< Iterations spent; the budget when not converged.

    /// The achieved convergence residuals of the returned iterate (schema
    /// 17): the true |E_n - E_{n-1}| in Hartree, and the RMS density change
    /// the SCF gate compared. They exist to QUALIFY `converged`, which is a
    /// bare Boolean that does not say what it stood on. nullopt means "not
    /// observed" - a result no SCF loop filled - never a fabricated zero
    /// (the certified_bound and spin_squared convention).
    ///
    /// Both gate legs are live on both paths, so a converged result has this
    /// below the run's [scf] energy_tolerance by construction (rhf.cpp's
    /// IsConverged requires both legs, and HfResult::achievedEnergyDelta
    /// records the same |E_n - E_{n-1}| that gate read). It was NOT bounded
    /// on RHF until the deferral was closed: the energy leg
    /// was inert there from the loop's first version, because the loop
    /// assigned its previous-iteration energy before the gate read it, so
    /// the leg compared the fresh energy against itself and the density leg
    /// decided every RHF stop on its own. A cached RHF result from before
    /// that fix is not comparable with one after it.
    std::optional<double> energyDeltaHartree;
    /// The RMS density change the gate compared (Frobenius/n) - on RHF, the
    /// leg that decided the stop on its own while the energy leg was inert,
    /// and now one of the two legs a stop must satisfy; below the run's
    /// [scf] density_tolerance by construction; the same null rule.
    std::optional<double> rmsDensityDelta;

    /// The linear-dependence removal's DISCLOSURE (schema 27): how many
    /// overlap directions the orthogonalizer removed because they fell below
    /// kOverlapEigenvalueFloorTolerance times the largest overlap eigenvalue.
    /// A system property - the basis set and the molecule - not a per-method
    /// switch, so RHF and UHF report the same count for the same inputs.
    ///
    /// Zero for every well-conditioned system, and a nonzero value is NOT a
    /// smaller calculation: the run happened in the reduced orthonormal space
    /// and was mapped back, so density, Fock and energy are all in the full AO
    /// dimension. It is reported because a silent truncation would be
    /// indistinguishable from a correct run (honoured, refused by name,
    /// or demoted with the disclosure) - and because a large diffuse basis is
    /// exactly where the removal is expected rather than exceptional.
    /// nullopt means "no SCF loop filled it" - never a fabricated zero.
    std::optional<std::size_t> numRemovedOverlapDirections;

    double totalEnergyHartree = 0.0; ///< Electronic plus nuclear repulsion.
    double electronicEnergyHartree = 0.0; ///< 1/2 Tr[D (H + F)].
    std::optional<double> spinSquared; ///< UHF spin-contamination diagnostic;
                                       ///< nullopt on RHF (not applicable).

    /// The population/moment block of the converged (or
    /// last-iterate) density; nullopt means "not computed", never "zero".
    std::optional<RunProperties> properties;

    /// The full-group labeling block: RHF's single channel (UHF's
    /// alpha). Absent when the stage did not run — a C1 molecule, the
    /// [symmetry] full_group = false switch, or an unrealizable group —
    /// never a fabricated C1.
    std::optional<RunSymmetry> symmetry;
    /// The UHF beta-channel labeling; absent on RHF (not applicable).
    std::optional<RunSymmetry> symmetryBeta;

    /// The alpha channel's symmetry-blocking disclosure (schema 32,
    /// RunSymmetryBlocking): the guard's decision and the numbers it
    /// rested on. Present exactly when the run WAS IN A POSITION TO BLOCK and
    /// took a decision; absent when it never asked — no basis set supplied, a
    /// C1 molecule, or an unrealizable group, which is the guard's own
    /// `kNotRequested`. The absence IS that statement, so no consumer can read
    /// "blocking was used" off a run that never requested it, and the
    /// vocabulary `action` carries has no `kNotRequested` word for the same
    /// reason.
    ///
    /// The value is the guard's own `action`, copied — never re-derived here
    /// from the two solve counts, which would be a second fact free to drift
    /// from the one the deciding code produced. The nesting is the
    /// per-channel pair: `symmetry_blocking` is written when EITHER channel
    /// disclosed, and inside it a channel that did not disclose is null (the
    /// num_removed_overlap_directions optional-to-null rule), so one spin's
    /// demotion still reaches the record while the other spin's Fock stayed
    /// adapted.
    std::optional<RunSymmetryBlocking> symmetryBlockingAlpha;
    /// The beta channel's disclosure; see symmetryBlockingAlpha.
    std::optional<RunSymmetryBlocking> symmetryBlockingBeta;

    /// The term_counters block of an instrumented ri_j run
    /// (RunTermCounters - the engine-emitted x/p3/g3/qx/gx per-run totals
    /// of the cost-table calibration). Absent on every other run (a
    /// non-ri_j kind or an uninstrumented ri_j run with no trace file) —
    /// never a fabricated block.
    std::optional<RunTermCounters> termCounters;

    /// The certified mixed-precision bound block (RunCertifiedBound):
    /// the fp32 lane's accumulated density-weighted kernel-bound sum, the
    /// run's certified-error audit quantity. Present exactly when the
    /// driver's seam observed at least one Fock build that actually
    /// carried the bound out-parameter — the direct family's machinery
    /// members on the RHF path, the only builders whose `BuildFock`
    /// exposes it — and absent everywhere it cannot be backed: the lean
    /// member (whose Schwarz-only builder computes no such bound at all),
    /// the RI-J/QFMM/GPU families, and the UHF branch — never a
    /// fabricated block. The seam's observation is gated by the same
    /// `requires` test that selects the out-parameter form, so a
    /// shorter-form builder's never-written 0.0 cannot reach the
    /// accumulator as a measurement. A present block whose bound is 0.0
    /// records the lane's TRUE zero: the routing criterion admitted no
    /// quartet at this preset (kTight's gate is 0.0), the lane was
    /// switched off, or no quartet qualified. It is never a statement
    /// that the run is exact.
    ///
    /// The block's enforcement members (schema 16 — `enforced`,
    /// `budget_ha`, `routed_ha`, `routed_quartets`, `fell_back_to_fp64`)
    /// carry the global budget check of the final observed call when
    /// the run asked for it ([method] enforce_certified_bound). An
    /// unenforced run's block carries them too - `enforced` false and the
    /// four quantities at their zero defaults, emitted rather than omitted,
    /// so the reader takes the flag as the answer and never infers the
    /// check's state from an absent key - and they are the enforcement's
    /// own there, never the delivered bound's. A fall-back reads as
    /// `fell_back_to_fp64: true` with a positive `routed_quartets` (the
    /// admissions the comparison refused) and a zero `last_call_ha` — the
    /// lane delivering nothing BECAUSE the budget refused it, which the
    /// reader must be able to tell from the lane delivering nothing on its
    /// own.
    std::optional<RunCertifiedBound> certifiedBound;

    /// The QFMM model block (schema 29, RunQfmmModel): what the
    /// composed-QFMM builder's Coulomb half actually ran - the geometry
    /// model, the separation test with its buffer, and the resolved theta.
    /// Present exactly when the run WIRED that builder (the RHF kQfmm arm
    /// and the composed-QFMM UHF runner, whether the family word was
    /// explicit or the size ladder's own tier resolved to it, and on the
    /// legacy null-budget path as well as the budget path), and absent on
    /// every other family - never a fabricated block. The presence rule
    /// follows the WIRED kind rather than the requested family word for the
    /// same reason `certified_bound` follows the observed call: a record
    /// that described the request instead of the build would be the
    /// mismatch this block exists to expose.
    ///
    /// Deliberately a member of the result rather than of
    /// `resources_resolved.mode_record`: that record is the Create-time
    /// rung decision and is ABSENT on the legacy null-budget path, which is
    /// a path a QFMM model can run on. The model is not a resource decision
    /// - it is a statement about which system the reported energy belongs
    /// to.
    std::optional<RunQfmmModel> qfmmModel;

    /// The resources_resolved block: the caps the run applied.
    RunResourcesResolved resourcesResolved;

    /// Wall times; total covers the whole run, scfLoop the SCF loop only.
    struct Timings {
        double totalMs = 0.0; ///< Whole-run wall time.
        double scfLoopMs = 0.0; ///< The Run*Scf call itself.
    } timingsMs; ///< The wall-time block.

    /// The unit the run's geometry was READ under (schema 28), serialized as
    /// "molecule_units": the RESOLVED interpretation the parser applied to the
    /// [molecule] atoms rows - kAngstrom for an absent `[molecule] units` key,
    /// which is the meaning every input file had before the key existed.
    ///
    /// Always present, because the alternative is a record that cannot answer
    /// the question it exists for: a reader holding only the JSON must be able
    /// to tell whether a geometry was read as Angstrom or Bohr without
    /// consulting the input file. The default reads "angstrom" rather than
    /// "not stated" on purpose - an unset member would put the ambiguity back,
    /// and the parser's own default IS Angstrom (a default is the
    /// system's judgement, and a judgement can be stated).
    ///
    /// It describes the INPUT, not the numbers: RunAtom coordinates are Bohr
    /// under both values (the parser converts once, at the boundary). Appended
    /// last so the aggregate's earlier fields keep their positions.
    CoordinateUnit moleculeUnits = CoordinateUnit::kAngstrom;

    /// The method word this run ran (schema 34), serialized as "method":
    /// "rhf" | "uhf" | "rks" | "uks" - the input's own `[method] type`
    /// vocabulary, through the same four words (ToString(MethodType, the
    /// inverse of ParseMethod)).
    ///
    /// Always present, and it is the run path the driver's ONE classifier
    /// resolved rather than the input enumerator re-read, so the word and the
    /// loop that produced the numbers cannot disagree. Before it, the record
    /// named the Fock builder (`resources_resolved.selection.builder`) and no
    /// physics at all: two runs differing by the whole RKS-vs-RHF difference
    /// were otherwise indistinguishable in the artifact a user keeps, and a
    /// comparison tool had to take the method from its own command line.
    /// Appended after moleculeUnits so the aggregate's earlier fields keep
    /// their positions.
    MethodType method = MethodType::kRhf;

    /// The RESOLVED functional name of a Kohn-Sham run (schema 34),
    /// serialized as "functional": the registry's own canonical spelling, not
    /// the file's string re-read - the driver resolves the input's
    /// `[method] functional` against the functional registry before anything
    /// is wired, and this is that lookup's answer.
    ///
    /// Present exactly when a density functional was named AND the run's path
    /// integrated it (the rks/uks words); ABSENT on every Hartree-Fock run,
    /// which names no functional - the input's own policy refuses the key
    /// there - so there is nothing to report and no fabricated word to report
    /// it with (the null-honesty rule: absence for a thing that was never
    /// asked, never a null or a placeholder). A consumer reading `method` and
    /// finding "rks"/"uks" therefore always finds this key beside it.
    std::optional<std::string> functional;

    /// The XC grid this run BUILT (schema 34), serialized as "xc_grid"
    /// (RunXcGrid): the six resolved settings the engine was created with.
    ///
    /// Present exactly when an XC grid engine was built - the same
    /// rks/uks-only condition `functional` carries, and the same absence on a
    /// Hartree-Fock run: no grid was involved in those numbers, so a block of
    /// defaults would be a claim about a grid that never existed. The block is
    /// what makes a run's quadrature answerable from the record alone, which
    /// matters because the grid is a physics choice: the same input at a
    /// different Lebedev size is a different energy, and before schema 34 the
    /// keys were compile-time defaults no input could reach and no record
    /// could name.
    std::optional<RunXcGrid> xcGrid;

    /// The RESOLVED builder selection in the orthogonal axis vocabulary
    /// (schema 35, RunBuilderAxes), serialized as "builder_axes": the family,
    /// the tier and the backend the run actually ran, with the deprecated
    /// `[method] fock_builder` word recorded as deprecated when the input used
    /// that key. The selection record above keeps the legacy word and the
    /// within-family member; this block states the same selection one axis at a
    /// time. Always present: every run resolves a builder, so the triple is
    /// always determinable and an absent block could only mean the driver forgot
    /// to fill it. Appended last so the aggregate's earlier fields keep their
    /// positions.
    RunBuilderAxes builderAxes;
};

/// Serializes a run result to the JSON schema (see the header note).
/// \param result The result to serialize.
/// \returns The pretty-printed JSON document; non-finite numbers appear
/// as the string markers "nan"/"inf"/"-inf" (see the header note).
/// \ingroup qcx-io
std::string SerializeRunResultJson(const RunResult& result);

} // namespace qcx::io
