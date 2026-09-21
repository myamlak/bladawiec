// The qcx run input schema: one TOML file describing a
// molecule, a basis, and an SCF method. The schema vocabulary is driven by
// the tree's own options structs - RhfOptions/UhfOptions, FockBuildOptions,
// the AccuracyPreset - not by an invented input language. The io module
// is a pure schema layer and stores what the file says, with ONE documented
// exception: the coordinate unit. `[molecule] units` names the unit the atom
// rows are written in (absent = Angstrom, the meaning every file had before
// the key existed), the parser resolves that key and applies the conversion
// exactly ONCE (parse_input.cpp's ParseAtom), and every RunAtom therefore
// holds Bohr - the internal convention of Molecule::Create - with
// RunMoleculeInput::coordinateUnit naming the unit it was read under. No
// consumer converts, and none can be handed the file's unit by accident: the
// pyscf-default mistake (Angstrom numbers read as Bohr) and its mirror (Bohr
// numbers fed to an Angstrom input file) were both this one mistake, made
// twice in opposite directions.
//
// Structural note: these structs are public aggregates - plain
// camelCase fields, no underscore prefix; the fields ARE the schema.

#pragma once

#include "qcx/integrals/accuracy.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace qcx::io {

/// \defgroup qcx-io Io module
/// The qcx run input schema and the result JSON: a pure schema
/// layer storing exactly what the TOML file says, plus the
/// serialization of the driver's JSON result.
/// \{

/// The unit a molecule's coordinates were READ under (the `[molecule] units`
/// key). The resolved interpretation, never a request left for a consumer to
/// apply: whatever the file spelled, `RunAtom` holds Bohr, and this names the
/// unit the numbers in the file were in. It is the run record's own value
/// (the `molecule_units` key), so a record is self-describing - a reader
/// holding the JSON can tell whether a geometry was read as Angstrom or Bohr
/// without consulting the input file.
/// \ingroup qcx-io
enum class CoordinateUnit {
    /// Angstrom (1e-10 m): the meaning of an absent `[molecule] units` key -
    /// the behaviour of every input file written before the key existed.
    kAngstrom,
    /// Bohr (a0): the internal convention. The file's numbers are already
    /// there, so the parser passes them through untouched (no multiply by
    /// one, no rounding, bit-exact by construction).
    kBohr
};

/// The unit word for one coordinate unit; the inverse of the parser mapping
/// (parse_input.cpp ParseCoordinateUnit). Unknown values never occur from the
/// parser; the fallback keeps diagnostics total anyway.
/// \param unit The coordinate unit.
/// \returns "angstrom" | "bohr".
/// \ingroup qcx-io
std::string_view ToString(CoordinateUnit unit) noexcept;

/// One atom as the parser stores it: an element symbol and three coordinates
/// in BOHR - the internal convention of Molecule::Create - whatever unit the
/// file spelled them in. The file's unit is resolved and applied exactly once
/// (parse_input.cpp's ParseAtom); the resolved unit is reported beside the
/// rows in RunMoleculeInput::coordinateUnit.
/// \ingroup qcx-io
struct RunAtom {
    std::string symbol; ///< IUPAC symbol, exact case.
    double x = 0.0; ///< X coordinate in Bohr (the file's unit applied once, at the parse).
    double y = 0.0; ///< Y coordinate in Bohr.
    double z = 0.0; ///< Z coordinate in Bohr.
};

/// The [molecule] block.
/// \ingroup qcx-io
struct RunMoleculeInput {
    int charge = 0; ///< Total electric charge.
    int multiplicity = 1; ///< Spin multiplicity 2S+1.
    std::vector<RunAtom> atoms; ///< Non-empty; validated by the parser. In Bohr.
    /// The unit the rows above were read under (the `[molecule] units` key;
    /// absent = kAngstrom), resolved by the parser and carried here so the
    /// run record can disclose it. It describes the FILE, not these fields:
    /// `atoms` is Bohr under either value. Appended last so the aggregate's
    /// earlier fields keep their positions.
    CoordinateUnit coordinateUnit = CoordinateUnit::kAngstrom;
};

/// The [basis] block. aux is optional: when the method needs an aux basis
/// and it is absent, the driver auto-selects an auxiliary basis.
/// \ingroup qcx-io
struct RunBasisInput {
    std::string orbital; ///< One of the bundled data/basis directory names.
    std::optional<std::string> aux; ///< Auxiliary basis, when explicit.
};

/// The `[grid]` block: the XC integration grid's build parameters,
/// mirroring `XcGridSettings` (`grid/include/qcx/grid/xc_grid_engine.hpp`)
/// field for field and in the same order.
///
/// Every key is optional and every default below is the engine's own
/// compile-time value, so an ABSENT block builds exactly the grid a run built
/// before the block existed: the defaults are the current behaviour stated,
/// not a new behaviour chosen. A present key is resolved HERE, once, by the
/// parser - the `[molecule] units` one-resolution rule - so the settings the
/// engine is created with and the grid the record discloses (`xc_grid`,
/// schema 34) are the same six numbers, and a run can never name a grid it
/// did not use.
///
/// The fields are plain rather than optional on purpose: an absent key and a
/// written default are the same request, and nothing downstream needs to tell
/// them apart (contrast `method.screeningTolerance`, where 0.0 is a different
/// PATH and the distinction is load-bearing).
/// \ingroup qcx-io
struct RunGridInput {
    std::size_t radialPoints = 75; ///< `radial_points`; radial points per atom.
    /// `angular_points`; a Lebedev size (qcx::grid::AngularGrid::kAvailableSizes).
    std::size_t angularPoints = 302;
    double alpha = 0.5; ///< `alpha`; the MHL radial mapping scale, Bohr.
    std::size_t radialExponent = 2; ///< `radial_exponent`; the MHL exponent m.
    double trimWeight = 1e-15; ///< `trim_weight`; |weight| below this is dropped.
    std::size_t blockTarget = 1024; ///< `block_target`; points per spatial block.
};

/// The SCF method family.
/// \ingroup qcx-io
enum class MethodType {
    kRhf, ///< Restricted Hartree-Fock (closed shell).
    kUhf, ///< Unrestricted Hartree-Fock.
    kRks, ///< Restricted Kohn-Sham (closed shell; a density functional, not HF).
    kUks, ///< Unrestricted Kohn-Sham.
};

/// The method word for one method type (schema 34); the inverse of the parser
/// mapping (parse_input.cpp ParseMethod), so the record's `method` key and the
/// input's `[method] type` key are one vocabulary - a consumer reads the same
/// four words on both sides. Unknown values never occur from the parser; the
/// fallback keeps diagnostics total anyway.
/// \param method The method family.
/// \returns "rhf" | "uhf" | "rks" | "uks".
/// \ingroup qcx-io
std::string_view ToString(MethodType method) noexcept;

/// The DEPRECATED (schema 35) FAMILY word of the key the axes replaced. The
/// builder vocabulary has a SECOND axis beside this one - the TIER, where the
/// working set of that family lives - and every word a document can write at
/// the [method] fock_builder key belongs to exactly one of the two (the
/// separation is named here because `direct` was being
/// read as one of its own tiers):
///
/// - FAMILY words are the enumerators below: `direct`, `ri_j_link`, `ri_jk`,
///   `qfmm`, `gpu` (plus `gpu_split`, which no [method] key spells). A family
///   word answers "computed on the fly, auxiliary-fitted, multipole-expanded
///   or on a device" - every family in this list computes its integrals on the
///   fly, so `direct` does NOT mean that, and reading it as a tier is the
///   confusion this note removes.
/// - TIER words are the four: `lean`, `in_memory`, `blocked`,
///   `disk`. Two of them ride this same fock_builder key, because they are the
///   within-family choices of the DIRECT family: `lean` (the Schwarz-only
///   member) and `in_memory` (kDirect's own tier - the word's alias, see
///   below). Neither grows an enumerator here (the kGpuSplit no-word
///   precedent): `lean` lands in RunMethodInput::leanDirect and `in_memory`
///   lands on kDirect. The other two name a rung INSIDE a family's own
///   builder and are spelled at their own keys: `blocked` is integrals'
///   RiFullFockRung::kBlocked, and `disk` is RiTensorMode::kDisk below (the
///   ri_j_link family's storage-module rung, the last rung of its ladder).
///
/// So a reader must not take a family word for a tier: `direct` names the
/// family, and the tier it runs is the in-memory one - kDirect's own note
/// states which tier it is, so this enum and RiTensorMode (the tier axis's own
/// enum) are the two places a word's axis can be read off.
///
/// v1 implements the wired subset documented on each enumerator; the un-wired
/// combinations are kUnimplemented from the driver, not silently substituted.
/// \ingroup qcx-io
enum class BuilderKind {
    /// The direct family's IN-MEMORY TIER: the batched-MD direct builder, the
    /// certified lane, its working set resident for the build - as against
    /// the same family's `lean` (Schwarz-only) tier, whose working set is the
    /// screened pair set. It is reached by the absent key above the lean
    /// ceiling, by the explicit family word `direct`, and by `in_memory` - the
    /// tier axis's own word for this tier, accepted at the [method]
    /// fock_builder key as an ALIAS of `direct`. One enumerator and one
    /// behaviour come out of the two spellings (parse_input.cpp's table maps
    /// both here), so no record and no refusal has to decide between them.
    kDirect,
    kRiJLink, ///< RI-J with direct exchange (RiJkFockBuilder).
    /// Full RI exchange (occ-RI-K). The BUILDER exists - `integrals/`
    /// RiFullFockBuilder - and so does the RUN PATH: the driver holds an
    /// `riJkBuilder` handle it reads the engine's term counters through, resolves
    /// this kind's auxiliary set as `FockBuilderKind::kRiJk` rather than
    /// falling back to the J-fit, and names the record's `builder_member`
    /// `kOccRiKMemberName` so the member the record states and the member the
    /// wiring executed cannot disagree. Two earlier readings this note carried are
    /// both superseded and neither may be re-derived from it: "no such path in
    /// code" (pre-builder) and "nothing wires this kind; the driver refuses the
    /// request by name"). What remains open is that kind's own
    /// quality and coverage work, which is a measurement
    /// sequence rather than a wiring gap - so this kind is REQUESTABLE and RUNS.
    kRiJk,
    kQfmm, ///< The octree multipole builder.
    kGpu, ///< The CUDA device builder; errors without a device.
    /// The gpu_split intra-build CPU+GPU batch-partition candidate. No [method]
    /// word in v1 - a heuristic
    /// pick only, never a user request - and no builder to execute: the
    /// kind exists so the driver's selection vocabulary can name the
    /// candidate and report it, while execution stays on the wired kinds.
    kGpuSplit,
};

/// The [method] word for one builder selection; the inverse of the parser
/// mapping (parse_input.cpp ParseBuilder). Unknown values never occur from
/// the parser; the fallback keeps diagnostics total anyway.
///
/// It returns the CANONICAL word for the kind, and the canonical word for
/// kDirect is "direct": `in_memory` - the tier axis's word for the same tier -
/// is an accepted INPUT alias that resolves to kDirect and is never produced
/// back, so every record and every refusal spells an in-memory-tier run
/// "direct" (the alias is a spelling of the request, not a second state the
/// run can be in).
/// \param builder The builder selection.
/// \returns "direct" | "ri_j_link" | "ri_jk" | "qfmm" | "gpu" |
/// "gpu_split".
/// \ingroup qcx-io
std::string_view ToString(BuilderKind builder) noexcept;

/// The canonical word of the accuracy preset: "kLoose" | "kNormal" |
/// "kTight" - the same words `[method] accuracy` accepts, so a record that
/// names a preset names the one the input could have written, and the parse and
/// the print cannot drift into two vocabularies.
///
/// It exists because the run record does not otherwise echo `[method] accuracy`
/// at all, and the approximated-exchange disclosure's bar is meaningless without
/// it: the bar is the preset's own per-atom budget
/// (`integrals RiExchangeErrorBudgetPerAtom`), and a bar a reader cannot tie to
/// a preset is the shape where a record states a value that silently
/// differs from what ran. Schema 31's `exchange_error.bar_preset` is the one
/// reader (the ri_jk disclosure plumbing).
/// \param preset The accuracy preset.
/// \returns The preset's input word.
/// \ingroup qcx-io
std::string_view ToString(qcx::integrals::AccuracyPreset preset) noexcept;

/// Whether this builder kind puts an auxiliary basis in effect. The RI-J
/// link is the one kind whose wiring fits its integrals through an
/// auxiliary set; every other kind builds from the orbital basis alone. The
/// driver reads this wherever an aux question is decided - the checkpoint
/// binding above all (the aux name enters the system fingerprint,
/// storage/fingerprint.hpp), and the wiring's own parse. The answer is
/// stated once, here beside the enum, under the exhaustive-switch guard: a
/// kind added to the enum without being placed in the switch stops the
/// BUILD, instead of inheriting an answer nobody chose.
/// \param builder The builder selection.
/// \returns True when the kind's wiring consumes an auxiliary basis.
/// \ingroup qcx-io
bool BuilderConsumesAux(BuilderKind builder) noexcept;

/// The RI tensor rung selection (the driver's disk-rung knob): the engine's
/// Create-time ladder selects among the IN-MEMORY rungs, and the disk rung is
/// entered only when a run asks for it - at this key (`"disk"`) or at the force
/// key that names the same rung more strongly (`[diagnostics] force_disk_ri`).
/// **Disk is never reached by a run that asks for nothing.** The ladder's own
/// last IN-MEMORY rung is the blocked-metric one; `disk` extends the ladder with
/// one further rung, and `auto` - the absent key - does not extend it at all.
///
/// These are TIER words, not family words (the two axes are stated at
/// BuilderKind): the key asks WHERE the RI tensor and its working set live -
/// `disk` names the storage-module disk-backed builder, and `auto` names no
/// tier at all (the engine's ladder decides). No word here selects a builder
/// family, and no family word is spelled here.
///
/// A rung selector selects a rung and nothing else: the FORCE lives in
/// `[diagnostics] force_disk_ri`, not in a
/// word here. A rung word re-used to mean "force" is one key naming two
/// mechanisms, which is what the key split separates (a key naming a
/// mechanism is refused or honoured by name; a preference may be disclosed).
/// \ingroup qcx-io
enum class RiTensorMode {
    kAuto, ///< No disk rung: the engine's in-memory ladder (fast -> light ->
           ///< blocked-metric) decides, and its refusal is the run's error.
           ///< The default when the key is absent.
    kDisk ///< Permits the disk rung (the storage-module disk-backed builder)
          ///< as the ladder's LAST rung: the driver still runs the composed
          ///< in-memory ladder first under the WorkspaceBudget and
          ///< engages the disk rung only on that ladder's estimate-time
          ///< refusal - the knob never makes disk a first choice. Requires
          ///< the [resources] budget path (the engine refuses only under a
          ///< budget; the legacy null-budget path cannot trigger the rung).
};

/// The [method] word for one RI tensor mode selection; the inverse of the
/// parser mapping (parse_input.cpp ParseRiTensorMode). Unknown values never
/// occur from the parser; the fallback keeps diagnostics total anyway.
/// \param mode The RI tensor mode selection.
/// \returns "auto" | "disk".
/// \ingroup qcx-io
std::string_view ToString(RiTensorMode mode) noexcept;

/// The default `[method] screening_tolerance`: the value a
/// Kohn-Sham run uses when the key is absent. It is the engine's measured
/// cheap route - at 1e-10 the density-weighted shell screening keeps ~85% of
/// the AO slots at a ~1e-12 Hartree energy error (the counted gate)
/// - while 0.0 selects the dense path for a cross-check. It lives here, at
/// the schema, because it is the documented default a reader of the input
/// format needs, not the driver's private choice.
/// \ingroup qcx-io
inline constexpr double kDefaultScreeningTolerance = 1e-10;

/// The [method] block.
/// \ingroup qcx-io
struct RunMethodInput {
    MethodType method = MethodType::kRhf; ///< The SCF method family.

    /// The `functional = "<name>"` request of a Kohn-Sham run:
    /// the density functional's NAME, deliberately UNRESOLVED here.
    /// io records the string the file carries and never learns which names
    /// exist; the driver resolves it against the functional registry and
    /// refuses an unknown one with the shipped set, so a new functional needs
    /// no change in this layer (the frozen-kernel contract's reasoning - names
    /// are the consumer's schema, enums drift with repo releases). The
    /// Hartree-Fock lanes REFUSE the key rather than ignore it (see
    /// validate_input.cpp's XcKeyPolicyFor): a `functional` on an rhf run
    /// means the author expected a DFT run, and running HF under that label is
    /// the failure the kRks/kUks refusal already exists to prevent.
    std::optional<std::string> functional;

    /// The `screening_tolerance` request in Hartree: 0.0 selects the dense
    /// path, a positive value the screened one. ABSENT means "use
    /// kDefaultScreeningTolerance". Absent and present are deliberately
    /// distinguishable - a plain `double` member with a default could not tell
    /// a written key from an omitted one, and the rhf/uhf refusal above needs
    /// exactly that distinction.
    std::optional<double> screeningTolerance;

    /// The Fock-builder selection; absent = auto (the direct-family
    /// default, selection_resolution.hpp). Explicit wins, the
    /// device-less GPU fallback and the v1 path restrictions apply - the
    /// driver resolves the effective builder before anything is wired.
    /// The explicit `fock_builder = "lean"` spelling never lands here: the
    /// lean member is within-family, so the builder slot keeps no word for
    /// it (the kGpuSplit precedent) and `leanDirect` below carries it.
    /// The tier axis's OTHER word, `in_memory`, DOES land here: it is the
    /// alias of `direct` (kDirect's own note - the same family, the same
    /// tier), so it fills this slot with kDirect and every consumer then
    /// reads one value for the two spellings.
    std::optional<BuilderKind> builder;
    /// The explicit `fock_builder = "lean"` request: the direct family's
    /// within-family lean (Schwarz-only) member, admitted at ANY size - at
    /// nBasis <= 1000 the absent key already selects it (the
    /// auto-lean default), above the ceiling this flag is the opt-in that
    /// reaches it. The builder slot stays absent for it (no family word
    /// was given), so the driver's resolution keeps resolving the direct
    /// family and the wiring picks the lean member on this flag. RHF-only:
    /// the UHF combination is refused up front (ValidateCombination).
    bool leanDirect = false;
    qcx::integrals::AccuracyPreset accuracy =
        qcx::integrals::AccuracyPreset::kNormal; ///< The accuracy preset.
    /// The explicit `enforce_certified_bound = true` request (the
    /// global budget enforcement): the certified fp32
    /// lane's routed bound sum is compared against the budget the accuracy
    /// preset derives, and a build that does not fit runs the fp64 lane
    /// alone (FockBuildOptions::enforceCertifiedBoundBudget). Off by
    /// default - the enforcement moves the fp32 lane's quartets to fp64,
    /// so it is a behaviour change an explicit builder request must not
    /// take silently.
    ///
    /// It is a CLAIM, not a preference: it asserts that the fp32 lane was
    /// certified against a budget, so there is no demotion available for it
    /// - the route either enforces it or the run is refused. One route
    /// enforces it, the RHF direct family's MACHINERY member (the budgeted
    /// builder's FastPath, whose other rungs - the LightPath and an engaged
    /// precision ladder - refuse by name from inside BuildFock), and every
    /// other route DROPS the field rather than enforcing it: the lean member
    /// (LeanFockBuildOptions carries no such field, so it is dropped by
    /// construction, and lean is the DEFAULT at nBasis <= 1000), the
    /// ri_j_link/QFMM/GPU families (RiEngineOptions carries no such field),
    /// and both UHF legs (the direct-UHF wiring never assigns the option).
    /// The driver therefore refuses the key by name - kUnimplemented,
    /// naming the resolved route - at the resolution point, before any
    /// builder is wired; the judgement is made on the resolved kind and the
    /// same within-family predicate the wiring reads, so the retargeted
    /// device-less gpu fallback (which lands on that machinery member)
    /// is admitted rather than refused. "Documented as a limitation" is not
    /// the disposition here: the field is enforced or refused, never
    /// quietly ignored. The record's certified_bound block reports the
    /// outcome, and it is ABSENT on every route that did not fold it - never
    /// a fabricated zero.
    bool enforceCertifiedBound = false;
    /// The explicit `force_certified_lane = true` request: engage the
    /// certified fp32 LANE itself,
    /// whatever the run's compute-profile probe measured. The probe still
    /// decides the DEFAULT - this key is the supported force-on override,
    /// and without this key the ON path is reachable only by
    /// injecting the probe's return value in source, so on a machine whose
    /// probe resolves OFF the pins could only ever assert the OFF
    /// behaviour.
    ///
    /// It is a CLAIM, not a preference: it asserts the lane was engaged, so
    /// there is no demotion available for it - the resolved route either
    /// carries the request into a live fp32 lane or the run is refused by
    /// name at the resolution point. The routes that carry it are
    /// the direct family's MACHINERY member (RHF and the direct-UHF
    /// machinery arm, whose coulomb and exchange halves are both
    /// FockBuildOptions) and nothing else: the lean member's
    /// LeanFockBuildOptions has no such field AND no fp32 lane at all, and
    /// the ri_j_link/QFMM/GPU families reach their own lanes (or none)
    /// without reading the request. `accuracy = "kTight"` is refused with
    /// them: MixedPrecisionThreshold(kTight) is 0.0 by construction,
    /// so the gate admits no quartet however the request reads.
    ///
    /// FALSE - the default, and an explicit `false` alike - is not a
    /// force-off: it leaves the lane's default to the probe, exactly as an
    /// absent key does. There is deliberately no force-off spelling here
    /// (naming one would have to mean something on the routes that run a
    /// lane the request cannot reach, which is a larger change than this
    /// key), so do not read `force_certified_lane = false` as "the lane is
    /// off".
    ///
    /// The override is VISIBLE: the run record's
    /// `resources_resolved.compute_profile.certified_lane_forced` is true
    /// exactly when this key asked for the lane, so a reader can never
    /// mistake a forced run's lane state for the machine's verdict (which
    /// the same block keeps reporting beside it).
    bool forceCertifiedLane = false;
    /// The QFMM well-separatedness override (the QfmmOptions.theta
    /// passthrough, the QFMM schema knobs):
    /// absent = the engine default, which resolves the accuracy preset's
    /// theta (ThetaForPreset); 0 = the preset's theta too; < 0 = the
    /// degenerate theta -> 0 gate (nothing well separated - everything near
    /// field, the bit-exact acceptance gate); > 0 = the explicit
    /// well-separatedness parameter. Finite only: nan and the infinities
    /// are rejected by the parser, each would otherwise silently bias the
    /// IsWellSeparated comparisons (nan/-inf classify nothing well
    /// separated, +inf everything) instead of failing. Consumed on the
    /// fock_builder = "qfmm" path only; the io layer stores what the file
    /// says.
    std::optional<double> theta;
    /// The QFMM multipole-order override (the QfmmOptions.lMult
    /// passthrough): absent = the engine default -1 (the accuracy
    /// preset's order, LMultForPreset); 0..8 = the explicit order - the
    /// kQfmmMaxLMult = 8 cap, the range QfmmJBuilder::Create
    /// enforces and this parser rejects past by name.
    std::optional<int> lMult;
    /// The QFMM octree leaf-size cap (the QfmmOptions.maxLeafSize
    /// passthrough): absent = the engine default 8; present must be
    /// >= 1 (the Create() contract).
    std::optional<std::size_t> maxLeafSize;
    /// The QFMM-vs-RI-J crossover override (the QfmmOptions.
    /// crossoverBasisFunctionCount passthrough): absent = the engine
    /// default -1, i.e. kQfmmCrossoverBasisFunctionCount (the
    /// placeholder until the measured value lands); present must be
    /// >= 0.
    std::optional<int> crossoverBasisFunctionCount;
    /// The RI tensor rung selection (the driver's disk-rung knob): "disk" permits the
    /// disk-backed rung as the in-memory ladder's last rung (RiTensorMode
    /// above); absent = auto. The FORCE is not a word here - it is the
    /// `[diagnostics] force_disk_ri` key below. Consumed on the
    /// fock_builder = "ri_j_link" path only; the io layer stores what the
    /// file says.
    std::optional<RiTensorMode> riTensorMode;
    /// The disk rung's chunk-size override (the DiskRiFockOptions.chunkBytes
    /// passthrough - the chunk-size floor of max(target,
    /// one aux shell)): absent = the engine default 256 MiB; present must be
    /// >= 1 (the Create() contract). Consumed on the fock_builder =
    /// "ri_j_link" path with ri_tensor_mode = "disk" or
    /// force_disk_ri = true only; the io layer stores what the file says.
    std::optional<std::size_t> riChunkBytes;
    /// The RI-J orbit-expansion opt-in: true
    /// makes the driver build BOTH point-group reductions and hand them to
    /// RiJkFockBuilder, so the 3c task grid is walked to one orbit
    /// representative per joint orbit and the rest of each orbit is expanded
    /// from the representative's block. Absent or false = the plain walk,
    /// which is what every run did before this key existed; the DRIVER
    /// supplies no reduction of its own, so the engine's own ON-by-default
    /// permission is inert here until this key asks for it. Consumed on the
    /// fock_builder = "ri_j_link" path only; the io layer stores what the
    /// file says. The run record's resources_resolved.ri_orbit_expansion
    /// pairs the request with what ran (engaged / not_requested /
    /// inert_trivial_group), because a key whose consumption is one branch
    /// deep owes the document its outcome.
    std::optional<bool> riOrbitExpansion;
    /// The disk-tier ERI store request (the engine-decorator
    /// seam's request surface, `EngineDecoratorFactory`): the path of the
    /// storage-module `CachedEriBatchEngine` store the run asks the driver
    /// to interpose between the Fock builder and the basis. A NON-EMPTY path
    /// IS the request - the `[scf] checkpoint_file` and `[properties]
    /// molden` spelling, where the path is the whole request and there is no
    /// separate boolean (the schema-surface rule: a path cannot be
    /// mistyped into another valid value). Empty (the default) and absent
    /// are the same not-requested state, and both leave every run
    /// bit-identical to before this key existed: no factory is installed, no
    /// store is opened, and the record carries no `eri_store` block.
    /// Consumed by the DRIVER, which is the one module allowed to link both
    /// sides (the module boundary): it constructs the
    /// decorator, owns it for the run, and reads its stats after the SCF.
    /// The io layer stores what the file says; the run record's
    /// resources_resolved.eri_store pairs the request with what actually
    /// ran, so a request that was demoted is disclosed rather than silently
    /// substituted.
    std::string eriCacheStore;
};

/// The [scf] block; every field optional with the SCF defaults.
/// \ingroup qcx-io
struct RunScfInput {
    int maxIterations = 100; ///< SCF iteration cap.
    double energyTolerance = 1e-8; ///< Energy convergence threshold (Eh).
    double densityTolerance = 1e-6; ///< Density-matrix convergence threshold.
    bool useDiis = true; ///< DIIS acceleration on by default.
    /// Diagnostics side-channel: when non-empty, the SCF loop appends one
    /// line per iteration to this file (trace_file) and the driver appends
    /// one line per main-SCF Fock-build call to `<file>.stats`. Empty (the
    /// default) keeps the zero-cost path - no files, no allocations, no
    /// numerical effect. Pure write-only diagnostics; nothing reads it.
    std::string traceFile;
    /// C12H26 discriminating-experiment diagnostics (RHF only): when
    /// non-empty, the RHF loop writes one binary stream - the magic
    /// "QXCDFDMP", the matrix dimension, and per-iteration (density,
    /// physical-Fock) pair records plus the overlap and core Hamiltonian
    /// once (format in scf_common.hpp's ScfDensityDumpWriter; consumed by
    /// tools/amf_density_invariants.py). Empty (the default) keeps the
    /// zero-cost path - no files, no allocations, no numerical effect
    /// (the bit-parity pins are absolute). UHF ignores it.
    std::string densityDumpFile;
    /// The last-iterate SCF checkpoint file (RHF only): when non-empty,
    /// the driver saves the SCF state - the spin-summed density, the DIIS
    /// history, the previous energies - through storage's
    /// SaveScfCheckpoint (HDF5, fingerprint- and checksum-verified on
    /// load) whenever the run exits without converging (the budget-exit
    /// case, so a later run can seed from the last iterate instead of
    /// re-wandering) and, when checkpointConverged is set, also on
    /// convergence. A stale file at the path is removed first (the
    /// checkpoint store is append-only by contract). Empty (the default)
    /// keeps the zero-cost path. UHF is kUnimplemented with this key.
    std::string checkpointFile;
    /// Whether the checkpoint (checkpointFile) is written also when the
    /// run converges; false (the default) writes it on the non-converged
    /// exit only. Meaningful only with checkpointFile.
    bool checkpointConverged = false;
};

/// The [resources] block (resource caps): the memory and thread
/// ceilings that steer the algorithm choice. Every field optional with the
/// documented defaults; the driver enforces them —
/// io is the pure schema layer and stores what the file says.
/// \ingroup qcx-io
struct RunResourcesInput {
    /// The hard process-memory ceiling in GiB, job-object enforced at
    /// driver start (the memory_gate pattern in-process). Default 16.0;
    /// 0 = no input cap (the documented escape hatch — the run takes
    /// whatever the machine has; the benchmark harness gate still applies
    /// to harness launches, and the two are independent by design).
    double memoryCapGiB = 16.0;
    /// The ceiling on the OpenMP team size: every parallel region runs
    /// with min(detected hardware threads, threadCap) and the
    /// fp64-half/fp32-full split applied to the clamped team. Default 0 =
    /// all detected hardware threads; 1 degrades to the serial path
    /// (bit-identical per its record).
    int threadCap = 0;
};

/// The [symmetry] block: the post-SCF full-group labeling stage
/// switch. Every field optional with the documented default; io is the
/// pure schema layer and stores what the file says.
/// \ingroup qcx-io
struct RunSymmetryInput {
    /// The full-group labeling stage runs by default (the stage is a
    /// pure a-posteriori classification — the SCF energy and state are
    /// untouched); false restores the pre-stage behavior bit-identically
    /// (the run output loses the symmetry block, nothing else changes).
    bool fullGroup = true;
};

/// The [memory_instrument] block: the per-term allocation
/// attribution opt-in. When enabled, the driver opens the attribution
/// window at run entry (AllocationInstrumentEnable) and closes it on
/// every exit path, so the run's tagged-allocation traffic lands in the
/// instrument's stats and — when trace_file is set — streams snapshot
/// rows to the trace. Purely observational: the instrument never changes
/// what the run allocates or computes (the bit-parity pins are
/// absolute). Every field optional with the documented default; io is
/// the pure schema layer and stores what the file says.
/// \ingroup qcx-io
struct RunMemoryInstrumentInput {
    /// The opt-in switch: false (the default) keeps the zero-cost path —
    /// the instrument stays disabled and the watchdog never starts.
    bool enabled = false;
    /// The watchdog's snapshot-interval in milliseconds: the row cadence
    /// while the instrumented run is active (the final row lands on close
    /// regardless). Must be >= 1.
    int snapshotIntervalMs = 250;
    /// The write-through trace-file path: the metadata header and the
    /// snapshot rows stream here, each row flushed, so the trace survives
    /// a cap kill. Empty (the default) enables the instrument without the
    /// watchdog — the stats window only, nothing written to disk.
    std::string traceFile;
};

/// The `[diagnostics]` block: the keys that
/// FORCE a mechanism a run would otherwise reach only by decision, so a
/// measurement cell can prove the mechanism engaged rather than being
/// silently swapped for the default. A force key is separate from every
/// selector it could have been smuggled into, and its name says FORCE: a
/// selector word re-used to mean "force" is one key naming two mechanisms
/// (the key split separates them by what the key NAMES).
///
/// Every field optional, default off; io is the pure schema layer and stores
/// what the file says - what a route can honour is the driver's
/// resolution point, not this layer's judgement.
/// \ingroup qcx-io
struct RunDiagnosticsInput {
    /// The RI disk rung's FORCE-ON request (replacing the retired
    /// `ri_tensor_mode = "forced_disk"` word): true makes the driver SKIP
    /// the composed in-memory ladder (no RiJkFockBuilder::Create, so no
    /// estimate-time refusal to fall back from) and construct the
    /// storage-module disk-backed builder directly, at any size and under
    /// any cap - the null-budget path included, where `ri_tensor_mode =
    /// "disk"` cannot reach the rung.
    ///
    /// It exists so a benchmark cell can measure the disk builder itself
    /// rather than an in-memory run wearing a disk label: at benchmark sizes
    /// the ladder fits, so `disk` alone never engages the store. It is a
    /// measurement/diagnostic override, NOT the production path - the
    /// ladder's own semantics are untouched and disk stays its LAST rung
    /// (the disk-algorithms principle: direct screened first,
    /// batched/blocked, recompute, disk LAST).
    ///
    /// `false` - explicit or absent - is NOT a force-off: it leaves every
    /// run's rung decision exactly where it was, which is the state every
    /// run had before this key existed. The run record states the override
    /// (resources_resolved.mode_record.forced_disk, which is true only on
    /// the forced route) and the requested-vs-ran pairing
    /// (resources_resolved.ri_tensor_mode, whose `forced` member names this
    /// key as the request's source).
    bool forceDiskRi = false;
};

/// The ESP point-set scheme of the [properties] block. io owns the schema
/// vocabulary (a pure schema layer cannot reference qcx::properties); the
/// driver maps this to the properties module's scheme on the way through.
/// \ingroup qcx-io
enum class EspFitScheme {
    kChelpg, ///< CHELPG cubic lattice [BrenemanWiberg1990].
    kMerzKollman ///< Merz-Kollman vdW shells [SinghKollman1984].
};

/// The [properties] block: which property analyses run after the SCF, on
/// top of the always-on population/moment block. Every
/// analysis is opt-in (all default off) - they run additional SCFs
/// (fukui: the N +- 1 species; hirshfeld: the promolecular SAD fragments;
/// nocv: the fragment SCFs) or heavy quadratures (hirshfeld/voronoi), so
/// the run stays cheap until an analysis is asked for.
///
/// The analyses that take the per-spin densities follow the properties
/// module's convention (RHF: alpha == beta == D/2; UHF: the per-spin
/// densities) - the driver passes exactly what the SCF seam produced.
/// \ingroup qcx-io
struct RunPropertiesInput {
    /// Hirshfeld (stockholder) charges [Hirshfeld1977]. Requires the
    /// promolecular SAD fragment densities, so the driver builds the SAD
    /// guess on demand; the v1 SAD atomic-multiplicity table limits the
    /// analysis to molecules with Z <= 10 (kUnimplemented beyond).
    bool hirshfeld = false;
    /// Voronoi (nearest-atom cell) charges [FonsecaGuerra2004].
    bool voronoi = false;
    /// ESP point-charge fit; nullopt (the default) runs no fit.
    std::optional<EspFitScheme> esp = std::nullopt;
    /// The EDDB delocalized-bond analysis [Szczepanik2014].
    bool eddb = false;
    /// The condensed Fukui indices [Parr1984]: the driver runs the N + 1
    /// (anion) and N - 1 (cation) species at the same geometry and basis
    /// with the direct-UHF path - the only UHF wiring in v1 - at the
    /// S +- 1/2 multiplicities (cation: M - 1 when M > 1, else 2; anion:
    /// M + 1), sharing the input's guess and SCF settings.
    bool fukui = false;
    /// The Nalewajski-Mrozek bond orders [Nalewajski1996]; the isolated-
    /// atom fragment SCFs limit the analysis to elements with Z <= 19
    /// (kUnimplemented beyond).
    bool nalewajski = false;
    /// The electron density at every nucleus: rho(R_A) per
    /// atom from the spin-summed AO density D = P_alpha + P_beta, in
    /// electrons/bohr^3 - a point evaluation, the Bader QTAIM
    /// prerequisite.
    bool densityAtNuclei = false;
    /// The Bader QTAIM critical points: the
    /// (3,-1) bond critical points of rho with their bond paths, plus the
    /// reported non-bond and unconverged critical points (the failed-
    /// search policy).  Newton searches on the spin-summed density; Bohr
    /// output.
    bool qtaim = false;
    /// The ETS-NOCV decomposition: the non-empty fragment
    /// partition, each group holding the atom indices of one fragment.
    /// The indices refer to the atom rows AS WRITTEN IN THE FILE (the
    /// driver maps them to the molecule's canonical atom order - the
    /// groups must be disjoint and cover every atom exactly once).
    /// Closed-shell only: rejected with kUnimplemented on UHF runs (the
    /// per-spin resolution assumes D_sigma = D/2, the closed-shell
    /// convention of qcx::properties::AnalyzeNocvEts). Empty (the
    /// default) runs no decomposition.
    std::vector<std::vector<std::size_t>> nocvFragments;
    /// The Molden-format F-file export path: the
    /// [Molden Format] [Atoms] (AU) [5D] [7F] [GTO] [MO] sections of the
    /// converged or last-iterate SCF result, written by
    /// io::WriteMoldenFile on the SCF branch. Empty (the default, and the
    /// meaning of an empty string in the input) runs no export.
    std::string molden;
};

/// The initial-guess selection. kGwh serves RHF and UHF (the RHF gwh
/// branch seeds the initial density; ValidateCombination accepts core and
/// gwh for RHF); kSad is UHF-only and needs the per-element fragment runs
/// the driver implements with its atomic-multiplicity table (Z <= 10);
/// kRestart is RHF-only in v1 and seeds from a last-iterate
/// checkpoint file (guessRestartPath), the run-restart read path.
/// \ingroup qcx-io
enum class GuessKind {
    kCore, ///< The METHOD'S DEFAULT START, not the P = 0 matrix. This word was
           ///< redefined: "`core` keeps its meaning as
           ///< 'the method's default start'"). On RHF the loop's own
           ///< start is the GWH guess - the zero start is a known-bad seed on
           ///< diffuse bases and that default replaced it - so a `core` request runs
           ///< GWH, and on UHF the loop's own Step-C default tier. The
           ///< word names the default; it does not name a zero density, and
           ///< a run whose start matters should say `gwh` explicitly.
    kGwh, ///< Generalized Wolfsberg-Helmholtz [Wolfsberg1952] (RHF and UHF).
    kSad, ///< Superposition of atomic densities (UHF).
    kRestart, ///< Seed from a saved SCF state (RHF; checkpoint file).
};

// ---------------------------------------------------------------------------
// THE BUILDER SELECTION'S ORTHOGONAL AXES.
//
// WHY THIS EXISTS. The `[method] fock_builder` key conflated two axes that
// answer different questions: a FAMILY answers *where the integrals come from*,
// a TIER answers *where that family's working set lives*. Two tier words rode
// that one key (the direct family's `lean`, and `in_memory` as an alias of
// `direct`), and two more tiers were reachable only as rungs inside a family's
// own builder. That is a fossil of separate decisions, and it becomes
// combinatorial the moment GPU DFT, RI, QFMM, disk and mixed precision all need
// orthogonal configuration. The key is now DEPRECATED and kept as an alias; the
// axis vocabulary below is its replacement.
//
// THE AXIS TABLE. One axis, one question, and every current spelling mapped to
// the combination it selects. The table is the contract this file, the parser
// and the run record are held to:
//
// | axis              | values                         | requested by                    |
// |-------------------|--------------------------------|---------------------------------|
// | integral_family   | direct, ri_j_link, ri_jk, qfmm | [builder] integral_family       |
// | storage_tier      | lean, in_memory, blocked, disk | [builder] storage_tier (2 of 4) |
// | execution_backend | cpu, gpu, gpu_split            | [builder] execution_backend     |
// | device            | host, cuda:<index>             | [builder] device                |
//
// The last row is the one axis that names no part of the SELECTION - it says
// where the resolved selection's kernels are required to execute, and it is the
// only axis a run may write alone without moving the family or the tier (the
// DeviceTarget doc states why). It is also the axis that cannot disagree with
// `execution_backend`: the two must agree, and a pair that does not is refused
// by name rather than resolved.
//
// LEGACY SPELLING -> COMBINATION (what `[method] fock_builder` meant):
//
// | legacy word | integral_family | storage_tier | execution_backend |
// |-------------|-----------------|--------------|-------------------|
// | absent      | direct          | lean <= 1000 basis functions, else in_memory | cpu |
// | direct      | direct          | in_memory    | cpu    |
// | in_memory   | direct          | in_memory    | cpu    |
// | lean        | direct          | lean         | cpu    |
// | ri_j_link   | ri_j_link       | in_memory    | cpu    |
// | ri_jk       | ri_jk           | in_memory    | cpu    |
// | qfmm        | qfmm            | in_memory    | cpu    |
// | gpu         | direct          | in_memory    | gpu    |
//
// The last row is the leak in its sharpest form: `gpu` reads as
// a family, and the builder's own header says what it is - GpuJkFockBuilder does
// "the same preparations as DirectJkFockBuilder::Create plus the device side",
// with "the CPU builder's screening semantics" and "its two-pass fp64 +
// certified-fp32 structure". A GPU builder whose algorithm is the direct
// family's is a BACKEND, not a family.
//
// WHERE EACH TIER VALUE IS SPELLED, and why not all four are one key's. `lean`
// and `in_memory` are the direct family's within-family choice, so they are
// `[builder] storage_tier`'s. `disk` is *already* an axis word with a home:
// `[method] ri_tensor_mode`, and the key split put
// the FORCE at `[diagnostics] force_disk_ri` rather than in a second spelling -
// so a `storage_tier = "disk"` spelling here would undo that separation and
// is REFUSED by name, with that key as the remedy (the `forced_disk` precedent,
// schema 22). `blocked` is integrals' RiFullFockRung::kBlocked, the full-RI
// transform accumulated in auxiliary shell ranges: it has no user request
// surface at all, and is named on the axis so the record can report it.
//
// VOCABULARY ONLY, THE kGpuSplit PRECEDENT PRESERVED. `gpu_split` is a value of
// execution_backend and remains a value no request can reach: the
// intra-build batch-partition candidate has no builder to execute, and a
// `[builder] execution_backend = "gpu_split"` request is refused by name rather
// than silently resolved (ValidateCombination refuses the kind itself for a
// programmatic caller). The axis names the candidate; execution stays on the
// wired kinds.

/// The builder selection's FAMILY axis: where the integrals come from. It says
/// nothing about where the working set lives (StorageTier) or where the kernels
/// execute (ExecutionBackend) - `direct` in particular names "no auxiliary fit
/// and no far-field expansion", not "in memory", and reading it as one of its own
/// tiers is the confusion the axis split removes.
/// \ingroup qcx-io
enum class IntegralFamily {
    /// No auxiliary fit and no expansion: each surviving quartet is evaluated
    /// against the orbital basis (the batched matrix-form McMurchie-Davidson
    /// engine).
    kDirect,
    /// The Coulomb half through an auxiliary fit, the exchange half exact and
    /// screened (`RiJkFockBuilder`).
    kRiJLink,
    /// BOTH halves through auxiliary fits - RI-J plus occ-RI-K from one
    /// metric-transformed 3-center tensor (`RiFullFockBuilder`).
    /// Its exchange half carries the fit's error, which the record DISCLOSES:
    /// rows built this way are a different result class from exact-kernel
    /// rows.
    kRiJk,
    /// Octree multipole expansion, far-field M2M/M2L/L2L (the multipole
    /// cap raised L=3 to L=8).
    kQfmm
};

/// The builder selection's TIER axis: where the selected family's working set
/// lives. Four values, one per tier the tree can actually run - see the axis
/// table above for which key spells each and why they are not all one key's.
/// \ingroup qcx-io
enum class StorageTier {
    /// The DIRECT family's Schwarz-only member: the screened pair set IS the
    /// working set. Requested by `[builder] storage_tier = "lean"` (direct family
    /// only), which is what the deprecated `fock_builder = "lean"` word meant.
    kLean,
    /// The working set resident for the build. Every family's default, and the
    /// tier the deprecated `direct` word named.
    kInMemory,
    /// The transform accumulated in auxiliary shell ranges
    /// (`RiFullFockRung::kBlocked`, the ri_jk family's own rung). NAMED here so
    /// the axis is total and the record can report it; it has no request surface.
    kBlocked,
    /// The storage-module disk-backed builder, the in-memory ladder's LAST rung.
    /// Spelled at its own key, `[method] ri_tensor_mode = "disk"`, never at
    /// the builder-selection keys.
    kDisk
};

/// The builder selection's BACKEND axis: where the family's kernels execute. It
/// is the axis the deprecated `gpu` word conflated with a family - the GPU
/// builder runs the DIRECT family's algorithm on the device (GpuJkFockBuilder,
/// its own header: the same preparations as DirectJkFockBuilder::Create plus the
/// device side, the CPU builder's screening semantics, its two-pass fp64 +
/// certified-fp32 structure).
/// \ingroup qcx-io
enum class ExecutionBackend {
    kCpu, ///< The CPU backend (OpenMP plus the SIMD tiers).
    /// The CUDA device backend. Refused by name in a build without
    /// CUDA; an explicit request with no device present is demoted to the ladder
    /// with a loud warning, never silently substituted.
    kGpu,
    /// The intra-build CPU+GPU batch-partition candidate. VOCABULARY ONLY,
    /// the kGpuSplit precedent: no builder to execute, so a request is refused by
    /// name (the parser at the axis key, ValidateCombination for a programmatic
    /// caller) while the axis keeps the candidate nameable.
    kGpuSplit
};

/// The builder selection's DEVICE axis: WHICH device a run requires its kernels
/// to execute on. It exists because the backend axis answers a different
/// question. `execution_backend = "gpu"` states the device CLASS - the run wants
/// the device path, whichever device that is - and before this axis there was no
/// key through which a run could name a PARTICULAR device, so a requirement of
/// that kind could only be left to be inferred from the builder kind the run
/// happened to resolve.
///
/// **The two name different things and the difference is the point.** The
/// builder kind says HOW the Fock build is done; this axis says WHERE the run
/// requires it to happen. A combination whose backend cannot supply the required
/// device is REFUSED BY NAME - never quietly executed elsewhere, which is the
/// rule the builder whitelist and the unwired-combination checks already follow.
/// The second surface for the same fact is safe here rather than contradictory
/// for that reason: it cannot disagree with the backend, because a disagreement
/// is a refusal and not a resolution.
///
/// **This axis does not restate the CUDA-absence refusal.** A build without the
/// CUDA backend refuses `execution_backend = "gpu"` by name already (the kGpu
/// doc above), so a `cuda:<index>` request reaches that refusal first and needs
/// no rule of its own here; the parser's acceptance matrix states only what the
/// two keys must agree about.
///
/// **A device requirement is not a builder selection.** Writing this key alone
/// does NOT fill the family or tier slots the way the `[builder]` axes do: with
/// the other axes absent the size ladder still decides the selection, and this
/// key only states where the run requires the result to be computed. That is the
/// whole reason it is a separate key rather than a fourth axis word on a key
/// that already conflates selection with placement.
/// \ingroup qcx-io
enum class DeviceTarget {
    /// The host: the CPU backend's own device, the one every run in a build
    /// without the CUDA backend executes on.
    kHost,
    /// A CUDA device, named by index in the selector.
    kCuda
};

/// One device requirement as the file wrote it: the target, and for kCuda the
/// index the selector named. `kHost` carries -1, which is not a device index -
/// the same absent-means-not-this-axis reading the sibling optionals use.
/// \ingroup qcx-io
struct RunDeviceRequest {
    /// The device class the requirement names.
    DeviceTarget target = DeviceTarget::kHost;
    /// The CUDA device index, or -1 on kHost.
    int index = -1;
};

/// The canonical selector text of one device requirement: `"host"` or
/// `"cuda:<index>"`. ONE spelling, produced in one place, so the parser's
/// accepted set, the input's stored form and the run record's member cannot
/// drift apart - and so the record's word is a word a user can write back into a
/// file (the `method` word's own rule).
/// \param request The requirement.
/// \returns The selector text.
/// \ingroup qcx-io
std::string DeviceSelectorText(const RunDeviceRequest& request);

/// One resolved builder selection, in the orthogonal vocabulary: the three axes
/// together are the whole selection, and no axis can be read off another.
///
/// This is the RESOLVED form. The deprecated `[method] fock_builder` key and the
/// `[builder]` axes are two SPELLINGS of one request, and both resolve here - so
/// a consumer reads one vocabulary and never has to know which key the file used
/// (the `in_memory` alias rule, generalized to the whole selection).
///
/// The TIER member reports the tier the BUILDER SELECTION resolved - `lean` or
/// `in_memory` (see `AxesOfBuilder`). The two tiers a rung key reaches are
/// disclosed by the rung blocks that own them rather than restated here, because
/// one fact has one home: `disk` by `resources_resolved.ri_tensor_mode` (whose
/// ladder decides at the engine's Create-time estimate, after this selection is
/// made) and `blocked` by the full-RI family's own rung record.
/// \ingroup qcx-io
struct BuilderAxes {
    IntegralFamily integralFamily = IntegralFamily::kDirect; ///< The family axis.
    StorageTier storageTier = StorageTier::kInMemory; ///< The RESOLVED tier, not the requested one.
    ExecutionBackend executionBackend = ExecutionBackend::kCpu; ///< The backend axis.
};

/// Whether two resolved selections name the same combination. Field-by-field
/// equality, so a new axis added to the struct is compared automatically rather
/// than silently ignored.
/// \param left One resolved selection.
/// \param right The other.
/// \returns True when all three axes are equal.
/// \ingroup qcx-io
bool operator==(const BuilderAxes& left, const BuilderAxes& right) noexcept;

/// The axis triple a builder kind and its within-family member resolve to - the
/// record's direction of the axis table, and the ONE place a kind's axes are
/// decided. Exhaustive by construction (the MachineryFamily rule in the driver):
/// a kind added to BuilderKind must be placed here or the build stops, instead
/// of inheriting an answer nobody chose.
/// \param kind The resolved builder kind.
/// \param leanMember True when the run wires the direct family's lean member
/// (ResolvedBuilderSelection::leanMember - the one statement of that choice).
/// \returns The combination the selection names.
/// \ingroup qcx-io
BuilderAxes AxesOfBuilder(BuilderKind kind, bool leanMember) noexcept;

/// The axis word for one family value; the inverse of the parser mapping
/// (parse_input.cpp ParseIntegralFamily), so the input's word and the record's
/// word are one vocabulary. Unknown values never occur from the parser; the
/// fallback keeps diagnostics total anyway.
/// \param family The integral family.
/// \returns "direct" | "ri_j_link" | "ri_jk" | "qfmm".
/// \ingroup qcx-io
std::string_view ToString(IntegralFamily family) noexcept;

/// The axis word for one storage tier; the inverse of the parser mapping
/// (parse_input.cpp ParseStorageTier).
/// \param tier The storage tier.
/// \returns "lean" | "in_memory" | "blocked" | "disk".
/// \ingroup qcx-io
std::string_view ToString(StorageTier tier) noexcept;

/// The axis word for one execution backend; the inverse of the parser mapping
/// (parse_input.cpp ParseExecutionBackend).
/// \param backend The execution backend.
/// \returns "cpu" | "gpu" | "gpu_split".
/// \ingroup qcx-io
std::string_view ToString(ExecutionBackend backend) noexcept;

/// The `[builder]` block: the orthogonal axis vocabulary that replaces the
/// deprecated `[method] fock_builder` key.
///
/// Every field is optional and an absent block changes nothing: an absent axis
/// resolves to the default the table above names, so a file that writes only
/// `[builder] execution_backend = "gpu"` selects the direct family's in-memory
/// tier on the device - which is exactly what the deprecated
/// `fock_builder = "gpu"` word meant.
///
/// ONE REQUEST, ONE KEY. The axes and the deprecated key must not both be
/// written: the parser refuses the pair by name (two keys naming one
/// mechanism), rather than picking a winner or silently checking agreement. The
/// deprecated key is accepted, never refused, and what it selected is recorded
/// as deprecated in `legacyFockBuilderWord` below and in the run record.
/// \ingroup qcx-io
struct RunBuilderInput {
    /// `integral_family`; absent = the family the deprecated key would have
    /// named, which for an absent key at all is the size ladder's (the direct
    /// family at every size, selection_resolution.hpp).
    std::optional<IntegralFamily> integralFamily;
    /// `storage_tier`; absent = in_memory, except that a `lean` request above
    /// the direct family's lean behaviour still resolves as the deprecated
    /// spelling did. `lean` on any family other than `direct`, and `disk` or
    /// `blocked` on any family, are REFUSED by name with the key that owns each
    /// (the axis table above).
    std::optional<StorageTier> storageTier;
    /// `execution_backend`; absent = cpu. `gpu_split` is refused by name: the
    /// candidate has no builder to execute (the kGpuSplit precedent).
    std::optional<ExecutionBackend> executionBackend;
    /// `device`; absent = no requirement stated, which leaves every run exactly
    /// where it was before the key existed. Present, it is the device the run
    /// REQUIRES (DeviceTarget above), and the two keys must AGREE: a `host`
    /// requirement beside `execution_backend = "gpu"`, or a `cuda:<index>`
    /// requirement beside `cpu` (or beside no backend word at all, whose
    /// resolution is cpu), is refused by name rather than resolved - the choice
    /// between them would have to elect one key as authoritative while silently
    /// leaving the other unread.
    ///
    /// It does NOT participate in the axes' selection: this key alone leaves the
    /// family, tier and backend at their own defaults, so the size ladder still
    /// decides the builder and the run only states where it requires the result.
    std::optional<RunDeviceRequest> device;
    /// The DEPRECATED spelling this run used, when it named its selection
    /// through `[method] fock_builder` rather than through the axes above: the
    /// exact word the file wrote ("direct", "ri_j_link", "ri_jk", "qfmm",
    /// "gpu", "lean" or the tier alias "in_memory"). Absent when the file wrote
    /// no builder key at all. It is the deprecation record, and the run record
    /// carries it so a reader can see which vocabulary the input used - a
    /// programmatic caller that fills `RunMethodInput::builder` directly leaves
    /// it absent by construction, which is why `explicit_builder` remains the
    /// member that answers "was a builder named at all".
    std::optional<std::string> legacyFockBuilderWord;
};

/// The full run description, parsed from one TOML file.
/// \ingroup qcx-io
struct RunInput {
    RunMoleculeInput molecule; ///< The [molecule] block.
    RunBasisInput basis; ///< The [basis] block.
    RunMethodInput method; ///< The [method] block.
    RunScfInput scf; ///< The [scf] block.
    RunResourcesInput resources; ///< The [resources] block.
    RunSymmetryInput symmetry; ///< The [symmetry] block.
    RunMemoryInstrumentInput memoryInstrument; ///< The [memory_instrument] block.
    GuessKind guess = GuessKind::kCore; ///< The initial-guess selection.
    RunPropertiesInput properties; ///< The [properties] block.
    /// The `[diagnostics]` block: the force-on keys (see
    /// RunDiagnosticsInput). All default off, so an absent block changes
    /// nothing.
    RunDiagnosticsInput diagnostics;
    /// The SCF checkpoint file the kRestart guess seeds from
    /// (guess.type = "restart"); empty with any other guess kind. Appended
    /// last so the aggregate's earlier fields keep their positions.
    std::string guessRestartPath;
    /// The `[grid]` block; absent when the file wrote none, and within a
    /// present block every key optional, the struct's own defaults above being
    /// the values an absent key resolves to. The optional around the whole
    /// block is a fact of its own, and the validation rule reads it: a grid
    /// key on a method word that builds no XC grid is REFUSED rather than
    /// ignored - the XcKeyPolicy posture its two neighbours `functional` and
    /// `screening_tolerance` already follow, because an rhf/uhf run that asked
    /// for a quadrature would otherwise run a different one and say nothing.
    /// Appended after guessRestartPath so the aggregate's earlier fields keep
    /// their positions.
    std::optional<RunGridInput> grid;
    /// The `[builder]` block: the orthogonal axis vocabulary. Every field
    /// optional; absent when the file wrote no
    /// `[builder]` key, in which case the selection came from the deprecated
    /// `[method] fock_builder` key or from the size ladder. Appended last so the
    /// aggregate's earlier fields keep their positions.
    RunBuilderInput builder;
};

/// \}

} // namespace qcx::io
