#pragma once

/// \file
/// The QFMM Coulomb-only Fock builder: J = near field + far field. The near
/// field is the existing DirectJkFockBuilder in its Coulomb-only mode
/// (FockBuildOptions::buildCoulombOnly), restricted to the octree near-field
/// pair-pair set (FockBuildOptions::restrictToPairPairs - the two-level
/// density screen decides which pairs are actually contracted, exactly as in
/// the unrestricted builder, so no second integral-contraction implementation
/// exists for the near field). The far field is the multipole accumulation
/// (the M2M/M2L/L2L passes of qfmm_multipole.hpp) at the preset's moment order, added on top
/// in the same 2J convention - the composed result is H + 2J(rho). The QFMM
/// builder computes Coulomb ONLY: the multipole far field has no meaningful
/// exchange analog, so K stays with the direct/RI exchange-only builders,
/// the RIJCOSX-style combination.
///
/// The octree, the interaction lists and the moment table are geometry-only
/// and built once in Create(); BuildFock only does density-dependent work
/// (the restricted near-field direct build and the density-weighted
/// far-field aggregation) - the DirectJkFockBuilder shape.
///
/// No Eigen in this header (integrals public headers keep Eigen
/// implementation-only - the stated CMake policy).

#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cstddef>
#include <memory>
#include <optional>

namespace qcx::integrals {

/// The pair-geometry model: which objects and radii the octree is
/// built over. kMidpointBound is the recorded form (the expansion centre is
/// the shell-pair midpoint and the bound covers every primitive product's
/// offset from it, which reaches molecule scale for tight x distant-diffuse
/// products); kProductBall is the corrected form (the centre and radius of
/// the pair's significant product distribution, the object the multipole
/// expansion is actually of).
/// \ingroup qcx-integrals
enum class QfmmExtentMode {
    kMidpointBound, ///< The recorded conservative midpoint-padded bound (retired, kept selectable).
    kProductBall, ///< The product-distribution ball (the corrected default since 2026-09-15).
};

/// The separation form: kWidthTheta is the centre-to-width test (distance x
/// theta against the node box half-widths) driven by the preset ladder's own
/// angles - the default classification; kSurfaceBall is the surface-to-surface
/// test (distance against the node charge radii plus a buffer).
/// \ingroup qcx-integrals
enum class QfmmSeparationMode {
    kWidthTheta, ///< The centre-to-width test (the default; its theta is the preset's own).
    kSurfaceBall, ///< The surface-to-surface ball test (selectable; the default 2026-09-15 to
                  ///< 2026-09-20).
};

/// The QFMM model record (the disclosure of what ran): the geometry
/// model, the separation test and its parameter, and the resolved theta of
/// one built QfmmJBuilder. It is what a reader needs to tell one classification
/// from another - the default build names the preset's own angle under the
/// centre-to-width form, which is a different pair set from the surface ball's
/// - the distinction a record that does not name the test leaves invisible.
/// This record reaches the run.
/// `QfmmHfFockBuilder::ModelRecord()` forwards it, the driver echoes it into
/// the run record's top-level `qfmm_model` block (`extent_model`,
/// `separation_mode`, `separation_k`, `theta`; run schema 29, 2026-09-15), and
/// that block is present exactly when the run wired the composed-QFMM builder.
/// The block is the record of the system's own judgement - none of the four is
/// an input key, so there is no request for the driver's knob rule to judge.
///
/// The budget fields below (errorAwareAdmission .. worstTruncationBound) are
/// added by this record and reach the run record's `qfmm_model` block beside
/// the four above (`error_aware_admission`, `far_field_budget`,
/// `per_interaction_budget`, `geometric_far_pair_count`,
/// `pairs_moved_to_near_field`, `fell_through_to_cap`,
/// `worst_truncation_bound`), so a run states whether its admission consulted
/// the budget and what the far field's own a-priori bound says about the
/// interactions it carried. `QfmmHfFockBuilder::ModelRecord()` forwards them
/// unchanged.
/// \ingroup qcx-integrals
struct QfmmModelRecord {
    QfmmExtentMode extentModel = QfmmExtentMode::kProductBall; ///< The geometry model used.
    QfmmSeparationMode separationMode = QfmmSeparationMode::kWidthTheta; ///< The test used.
    double separationK = 0.0; ///< The surface-ball buffer (negative = the degenerate gate).
    double theta = 0.0; ///< The resolved theta (kWidthTheta's parameter).

    /// Whether the error arm was armed - admission consulted the budget as well
    /// as the geometry - or ran the geometric criterion alone. False on every
    /// build that did not ask for the arm (QfmmOptions::enforceFarFieldBudget),
    /// which is the default at every preset.
    bool errorAwareAdmission = false;
    double farFieldBudget = 0.0; ///< The preset budget the far field was held to (Eh).
    double perInteractionBudget = 0.0; ///< The budget split over the geometrically admissible
                                       ///< pairs - the share each far interaction is allowed.
    std::size_t geometricFarPairCount = 0; ///< Far pairs the geometric criterion found (the
                                           ///< split's denominator, and what the count would be
                                           ///< without the error arm).
    std::size_t pairsMovedToNearField = 0; ///< Admitted geometrically, refused by the budget.
    /// Interactions whose a-priori truncation bound at the order cap exceeds
    /// their share of the preset budget: the fall-through that used to be
    /// silent, and the number that says whether the bound CERTIFIES the run.
    /// The measured error is a different question and may still be inside the
    /// budget (a worst-case-over-distributions bound is not a prediction) - a
    /// nonzero count here with an inside-budget error is the honest record of
    /// exactly that. Zero while errorAwareAdmission is true: the arm admits
    /// only what the bound certifies.
    std::size_t fellThroughToCap = 0;
    /// The worst interaction's truncation bound at the cap (0 when there are no
    /// far pairs). Divide it by perInteractionBudget for how far outside the
    /// budget the worst interaction sits.
    double worstTruncationBound = 0.0;
};

/// QFMM build settings (the near-field direct-builder pass-throughs the
/// bit-exact acceptance gate requires).
/// \ingroup qcx-integrals
struct QfmmOptions {
    AccuracyPreset accuracy = AccuracyPreset::kNormal; ///< The accuracy preset: the
                                                       ///< near-field screening thresholds and,
                                                       ///< unless overridden below, the
                                                       ///< well-separatedness parameter and the
                                                       ///< moment order (ThetaForPreset /
                                                       ///< LMultForPreset).
    double theta = 0.0; ///< The well-separatedness angle: 0 is "not given" and means the
                        ///< preset's OWN angle (ThetaForPreset: 0.45 loose, 0.3 normal, 0.0
                        ///< tight), which is the criterion the default form (separationMode
                        ///< kWidthTheta) classifies with - so the ladder's declared aim is
                        ///< the aim the build runs; > 0 overrides it at that value, which is
                        ///< the historic contract kept intact (no written accuracy control is
                        ///< silently ignored); < 0 is the degenerate gate under EITHER form
                        ///< (nothing well-separated - everything near field, the bit-exact
                        ///< acceptance gate). theta is ignored by kSurfaceBall
                        ///< only when it is 0. NaN is rejected by Create.
    int lMult = -1; ///< The multipole-order override: -1 uses LMultForPreset(accuracy);
                    ///< 0..kQfmmMaxLMult (0..8) uses the value directly; anything else is
                    ///< rejected by Create.
    bool useAdaptiveOrder = true; ///< The adaptive per-interaction multipole order:
                                  ///< when true, each far-field interaction uses the minimal order
                                  ///< whose geometric bound (source radius / distance)^(L+1)
                                  ///< is within the preset budget's per-interaction share
                                  ///< (QfmmBudgetForPreset / far-pair count), capped at the
                                  ///< resolved lMult above - the selector only lowers L where
                                  ///< the bound proves it safe, so the preset ladder stays
                                  ///< the accuracy contract. The source radius is the NODE's
                                  ///< charge radius - the covering ball of the charge the
                                  ///< moments are of, and the same quantity admission reads -
                                  ///< not the box circumradius sqrt(3) * halfWidth it used to
                                  ///< take, which over-states it by the box's own slack. An
                                  ///< interaction the bound cannot bring within the share runs
                                  ///< at the cap and is COUNTED (QfmmModelRecord::
                                  ///< fellThroughToCap), so a budget miss is a record entry
                                  ///< rather than an inference from an accuracy miss. When
                                  ///< false, every interaction uses the resolved lMult (the
                                  ///< fixed-order path); the count is still reported.
    /// Error-aware admission: when true, a node pair is treated as far only if
    /// some degree up to the resolved order keeps the a-priori multipole
    /// truncation bound within that interaction's share of the preset budget; a
    /// pair that fails descends into the near field, where it is computed
    /// exactly. The share is the preset budget divided over the geometrically
    /// admissible pairs, the model's own interaction count. The arm sits inside
    /// the classification predicate, so it applies to whichever separation form
    /// is in use - with this on, admission and order selection read one bound
    /// and cannot disagree, and QfmmModelRecord::fellThroughToCap is zero.
    ///
    /// DECIDED, off by default, and the decision is measured. The truncation
    /// bound is a worst case over distributions: it asks for a source-radius
    /// ratio of 0.011 at the normal preset's L = 5 and 0.048 at the cap L = 8
    /// (a centre distance of 94x and 21x the source radius), while the pairs any
    /// usable angle admits sit far inside that - measured on C24H50, the 33,404
    /// pairs the preset's own angle admits run 0.07..0.27 and the 69,265 the
    /// surface ball at k = 1 admits run 0.12..0.47. So with the angle wired the
    /// arm declares every admitted pair unrepresentable: measured, it moves all
    /// 33,404 to the near field and the far field carries nothing (C12H26: all
    /// 517, C6H14: all 16). That is the near-field direct build - exact, and
    /// inside the budget by having nothing left to miss - but it is also exactly
    /// what wiring the preset's own angle exists to avoid, so it is not the
    /// default. The default instead is the preset's own aim (which measures
    /// inside the preset budget with an order of magnitude to spare, see
    /// separationMode) with the a-priori bound's verdict on every interaction
    /// still reported through QfmmModelRecord. Turn this on for a build that
    /// carries only what the bound certifies - measured, an empty far field on
    /// every fixture measured.
    bool enforceFarFieldBudget = false;
    std::size_t maxLeafSize = 8; ///< The octree leaf-size cap (BuildQfmmTree), must be >= 1.
    /// The pair-geometry model (the internal QfmmExtentModel's public
    /// vocabulary): the product-distribution ball the multipole expansion is
    /// actually of is the DEFAULT since the owner ruling of 2026-09-15
    /// ("enable the corrected path now, both keys on by default"); the
    /// recorded midpoint-padded extent is kept, retired, for reproducibility
    /// of pre-ruling runs. EVIDENCE for the default (C24H50/def2-SVP, 586 BF,
    /// kNormal, leaf cap 8, k = 1): the far field carries 37.7% of the direct
    /// CSR and 41.7% of its function quartets (7.43e8 saved) for 1.65e8
    /// multipole blocks - 4.50 quartets saved per block, rising with size
    /// (1.59 at 170 BF) - at a 1.1e-8 Ha Fock error against the exact
    /// near-field-only build. The retired midpoint bound measured
    /// 1.60/12.47/29.62 Bohr min/med/max against the corrected
    /// 0.086/4.45/8.69, and at the operating theta it removed 36 of
    /// 122,319,961 CSR classes: a far field that was not engaged.
    QfmmExtentMode extentModel = QfmmExtentMode::kProductBall;
    /// The separation form (the internal QfmmSeparationTest's public
    /// vocabulary). The DEFAULT is kWidthTheta, the centre-to-width test
    /// distance x theta >= wA + wB - and its theta is the PRESET'S OWN ANGLE
    /// (ThetaForPreset: 0.45 loose, 0.3 normal, 0.0 tight), so the ladder's
    /// declared aim is the aim the classification runs. MEASURED at C24H50 /
    /// STO-3G / normal / product-ball extents / cap L = 5: the preset's own
    /// 0.3 keeps 33,404 of the pairs a looser test would admit and measures
    /// 1.0006e-08 Eh against the direct build - the 1e-7 budget with an order
    /// of magnitude to spare - while the same fixture on the surface ball at
    /// k = 1 admits 69,265 pairs and measures 1.0994e-04, 1100x outside that
    /// budget. The angle is also what makes the LADDER mean something on the
    /// default path: kTight's 0.0 is the degenerate gate (its near-field-only
    /// identity, kept), kNormal's 0.3 is the rung above it. The surface ball is
    /// kept and still selectable - set this to kSurfaceBall and set separationK -
    /// for callers who want that geometric form.
    QfmmSeparationMode separationMode = QfmmSeparationMode::kWidthTheta;
    /// The kSurfaceBall buffer, in units of max(rA, rB): 0 is the bare
    /// touching test, positive is a buffer, and a NEGATIVE value is the
    /// degenerate gate (nothing well separated - the theta -> 0 analogue, so
    /// the far field is empty and the build is the exact restricted near-field
    /// direct build). Read only when separationMode is kSurfaceBall (the
    /// default form classifies by theta, and this field does not enter it).
    /// The DEFAULT 1.0 was the measured safe rung of that form (1.1e-8 Ha Fock
    /// error at C12H26/STO-3G against the exact near-field-only build), where
    /// the bare touching test (0.0) measures 2.5e-5 - outside the kNormal
    /// budget; on C24H50 it measures 1.0994e-04, outside that budget too, which
    /// is why it is no longer the default form - the field is read only when
    /// that form is named. An explicit theta < 0 still selects the degenerate
    /// gate under EITHER form (nothing well separated - the far field empty and
    /// the build the exact restricted near-field direct build).
    ///
    /// MEASURED, and the reason the buffer is a rung rather than the bound's
    /// own demand: the a-priori truncation bound needs a charge-radius ratio of
    /// 0.011 at the kNormal order (L = 5) and 0.048 at the cap (L = 8) to cover
    /// the preset budget's per-interaction share, which is a centre distance of
    /// 94 and 21 times the source radius - a buffer of k ~ 90 and k ~ 20. The
    /// admitted pairs sit at ratios 0.123..0.471 (C24H50/STO-3G, kNormal,
    /// product-ball extents, measured), i.e. 2.5x to 44x looser than the bound
    /// demands. No measured rung lives anywhere near k = 20, so the buffer is
    /// calibrated by what the composed error actually does, and the bound's own
    /// verdict on a run - how far outside the share the worst interaction sits -
    /// is reported through QfmmModelRecord instead of being gated on. Gating on
    /// it is QfmmOptions::enforceFarFieldBudget, off by default.
    double separationK = 1.0;
    bool useDensityScreening = true; ///< Density-weighted pair screening per iteration, passed
                                     ///< through to the near-field direct builder's
                                     ///< FockBuildOptions.
    bool useCertifiedMixedPrecision = true; ///< Certified mixed precision in the near field,
                                            ///< passed through to the near-field direct builder's
                                            ///< FockBuildOptions.
    std::size_t maxParallelChunks = 0; ///< The whole-builder parallel chunk override: the
                                       ///< near field passes it through to the nested direct
                                       ///< builder's FockBuildOptions, and every far-field pass
                                       ///< of qfmm_multipole.hpp resolves it on its own task
                                       ///< count (0 = auto; 1 = the serial determinism pin the
                                       ///< bit-exact theta -> 0 gate needs - the chunked
                                       ///< combine order is thread-schedule dependent;
                                       ///< >= 2 a fixed split).
    int crossoverBasisFunctionCount = -1; ///< The QFMM-vs-RI-J crossover override: -1 uses
                                          ///< kQfmmCrossoverBasisFunctionCount (the unmeasured
                                          ///< placeholder); >= 0 overrides it (the decision
                                          ///< function treats any basis-function count at or
                                          ///< above the threshold as "use QFMM").
    /// The adaptive-memory workspace budget (fock_build.hpp
    /// FockBuildOptions::workspaceBudget - same contract): when set, Create
    /// decides the build mode from the Create-time footprint estimate (the
    /// near-field builder's estimate plus the outer store: tree, moments,
    /// far-field list and bitset) against the budget's remaining bytes. The
    /// budget is shared by pointer with the nested near-field direct builder
    /// (its Create-time reservation orders after this one - nesting order =
    /// reservation order); the pointer is MUTABLE (Reserve charges the
    /// cumulative counter). Null (the default) keeps the legacy behavior
    /// exactly. The pointer must outlive the Create call.
    qcx::memory::WorkspaceBudget* workspaceBudget = nullptr;
    /// The point-group reduction, REFUSED by name rather than honoured (the
    /// declared contract: a user-selectable surface is honoured, refused with its
    /// condition named, or demoted with the disclosure - never silently
    /// substituted). This builder does not implement the reduction at all, and
    /// the field exists so that the request can be refused instead of quietly
    /// dropped: the near field that carries the work is this builder's nested
    /// DirectJkFockBuilder - so the lever, if a later increment takes it,
    /// lives on that builder's own options (FockBuildOptions::symmetryReduction)
    /// at scales where its class path's Create-time table fits. Setting this
    /// field returns kUnimplemented (QfmmJBuilder::Create).
    ///
    /// This entry used to add that there was almost nothing for a reduction to
    /// serve, quoting the RETIRED midpoint/width separation criterion's far
    /// field - 36 of 122,319,961 screened entries at C24H50/def2-SVP, 0.00003%,
    /// rising to 0.4% at C24H50/STO-3G, with the near field carrying the other
    /// 99.99997%. The 2026-09-15 default flip retired that model:
    /// under the surface-ball criterion the same fixture's far field carries
    /// 37.72% of the CSR classes and 41.7% of the function quartets, so the
    /// "almost nothing to serve" framing is false and is no longer made. The
    /// refusal stands on the missing implementation, which is unaffected by the
    /// flip - what changed is that it now denies a REAL reduction.
    const SymmetryReduction* symmetryReduction = nullptr;
};

/// The QFMM-vs-RI-J crossover threshold: the basis-function count at or
/// above which the QFMM path is recommended over RI-J. QFMM only pays off
/// once the octree near/far split reduces work relative to RI-J's own O(N)
/// 3-center cost - below the threshold RI-J is cheaper and simpler.
/// UNMEASURED placeholder: no QFMM-vs-RI-J benchmark exists yet, so
/// this is a round-number stand-in, not a validated crossover.
/// \ingroup qcx-integrals
inline constexpr std::size_t kQfmmCrossoverBasisFunctionCount = 500;

/// The QFMM-vs-RI-J crossover decision (whether the QFMM path is
/// recommended for a system of \p basisFunctionCount basis functions): the
/// threshold is
/// kQfmmCrossoverBasisFunctionCount unless QfmmOptions::
/// crossoverBasisFunctionCount is >= 0, in which case the override wins.
/// \param basisFunctionCount The system's basis-function count.
/// \param options The QFMM settings (the crossover override, if any).
/// \returns True when the count is at or above the threshold.
/// \ingroup qcx-integrals
inline bool RecommendQfmmOverRiJ(std::size_t basisFunctionCount,
                                 const QfmmOptions& options) noexcept {
    const std::size_t threshold =
        options.crossoverBasisFunctionCount >= 0
            ? static_cast<std::size_t>(options.crossoverBasisFunctionCount)
            : kQfmmCrossoverBasisFunctionCount;
    return basisFunctionCount >= threshold;
}

/// Builds F(D) = H + 2J(rho) for closed-shell RHF from the QFMM octree:
/// the octree near field through the restricted direct Coulomb builder, the
/// far field through the multipole accumulation at the preset's moment order.
/// The input density is the SPATIAL closed-shell density rho = D/2, the
/// DirectJkFockBuilder convention (the scf FockBuilderFn seam's adapters
/// scale by 1/2).
/// \ingroup qcx-integrals
class QfmmJBuilder {
public:
    /// Prepares the builder: flattens the pairs, builds the pair data, the
    /// octree, the interaction lists (theta resolved per QfmmOptions), the
    /// moment table, and the restricted near-field direct builder - all
    /// one-time, geometry-only work.
    /// \param molecule Molecule providing the atom coordinates (Bohr).
    /// \param basisSet Basis set; every shell must satisfy l <= kMaxEngineL
    /// (kUnimplemented otherwise, from the near-field builder).
    /// \param coreHamiltonian H = T + V, host-canonical rank-2 tensor with
    /// shape {n, n}.
    /// \param options QFMM build settings; theta must not be NaN, lMult must
    /// be -1 or in 0..kQfmmMaxLMult, maxLeafSize must be >= 1.
    /// \returns The builder, or an Error.
    static qcx::Result<QfmmJBuilder> Create(
        const qcx::molecule::Molecule& molecule,
        const qcx::basisset::BasisSet& basisSet,
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
        const QfmmOptions& options = {});

    /// Builds F(D) = H + 2J(rho) for the given closed-shell density: the
    /// near-field direct build (restricted to the octree near-field pair-pair
    /// set) plus the far-field multipole accumulation (density-weighted leaf
    /// moments, M2M, M2L + L2L, and the 2J accumulation - the far field is
    /// empty at the theta -> 0 gate, so that limit is exactly the restricted
    /// direct build).
    /// \param density The SPATIAL closed-shell density rho = D/2, host-
    /// canonical rank-2 tensor with shape {n, n}. Must be SYMMETRIC (the
    /// direct builder's documented precondition - the canonical-pair
    /// contractions read both orientations of every unordered pair block).
    /// \returns The Fock matrix as a rank-2 tensor, or an Error.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildFock(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density) const;

    /// The number of far-field pair-pair interactions in the built octree
    /// (the liveness flag): nonzero means the multipole far field is
    /// live for this geometry and theta, zero means the build degenerates
    /// to the near-field direct build (the vacuous-ladder trap).
    /// \returns The far-field pair count from the last successful Create().
    std::size_t FarFieldPairCount() const noexcept;

    /// The Create-time mode record (fock_build.hpp FockModeInfo); nullopt
    /// when no workspace budget was given (the legacy path - no decision).
    /// \returns The mode record, or nullopt on the legacy path.
    const std::optional<FockModeInfo>& ModeInfo() const noexcept;

    /// What this build actually ran, stated (the disclosure applied to a default): the
    /// geometry model, the separation test and its parameter, and the resolved
    /// theta. Always available (unlike ModeInfo, which the legacy path leaves
    /// empty), because the default itself is what a reader must be able to see -
    /// a far field built over product distributions classified at a preset angle
    /// is a different system from one built over the recorded midpoint extents
    /// with the surface ball, and the run record must say which one ran.
    /// \returns The model record of the last successful Create().
    QfmmModelRecord ModelRecord() const noexcept;

private:
    // The implementation state lives in the .cpp (Eigen stays
    // implementation-only).
    struct State;
    explicit QfmmJBuilder(std::shared_ptr<const State> state);
    std::shared_ptr<const State> _state;
};

} // namespace qcx::integrals
