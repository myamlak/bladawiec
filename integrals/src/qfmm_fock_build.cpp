// The QFMM Coulomb-only Fock builder: per BuildFock the
// near-field direct build (the existing DirectJkFockBuilder in Coulomb-only
// mode, restricted to the octree near-field pair-pair set - the two-level
// density screen decides which pairs are actually contracted) plus the far-field
// multipole accumulation (the density-weighted leaf moments, the M2M and
// M2L + L2L passes and the 2J accumulation of qfmm_multipole.hpp, in the
// direct builder's own convention so the two halves compose). The octree,
// the interaction lists and the moment table are geometry-only and built
// once in Create(); BuildFock only does density-dependent work.

#include "qcx/integrals/qfmm_fock_build.hpp"

#include "internal/fock_screen.hpp"
#include "internal/footprint.hpp"
#include "internal/md_batch.hpp"
#include "internal/qfmm_geometry.hpp"
#include "internal/qfmm_multipole.hpp"
#include "internal/qfmm_tables_gen.hpp"
#include "internal/qfmm_tree.hpp"
#include "internal/tensor_eigen_bridge.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace qcx::integrals {

struct QfmmJBuilder::State {
    ShellPairList _pairList;
    std::vector<internal::MdPairData> _pairStore;
    std::vector<internal::QfmmPairGeometry> _geometries;
    internal::QfmmTreeBuildResult _tree;
    // The well-separated node pairs of the resolved theta, built once: the
    // near/far classification is geometry-only, density-independent.
    std::vector<std::pair<std::size_t, std::size_t>> _farFieldPairs;
    internal::QfmmMomentTable _momentTable;
    // The order bands: the adaptive per-far-pair truncation orders (the M2L caps) and
    // the per-node orders they imply (the aggregation/L2L/accumulation caps
    // and the moment-table stride; -1 marks a near-field-only node/pair,
    // which holds zero moment blocks). Geometry-only, built once.
    std::vector<int> _farPairOrders;
    std::vector<int> _nodeOrders;
    // The far-field chunk override: the QfmmOptions::maxParallelChunks value, copied at Create for
    // BuildFock - every far-field pass of qfmm_multipole.hpp resolves it on
    // its own task count (0 = auto per pass, 1 = the serial bit-exact pin,
    // >= 2 a fixed split), the same 0/1/>=2 contract the near-field
    // builder's FockBuildOptions carries. The far field is approximate by
    // construction, so the chunked fp grouping of the M2L adds (the one
    // chunked far-field reduction) is budget-irrelevant, but its join is in
    // fixed chunk order - bit-reproducible run to run at every chunk count.
    std::size_t _maxParallelChunks = 0;
    // The near-field half: the restricted direct Coulomb builder (H + 2J_near
    // per BuildFock call). Optional because DirectJkFockBuilder is not
    // default-constructible; always engaged after Create().
    std::optional<DirectJkFockBuilder> _nearFieldBuilder;
    // The adaptive-memory Create-time mode record (fock_build.hpp
    // FockModeInfo); nullopt on the legacy path (no decision - ModeInfo()).
    std::optional<FockModeInfo> _modeInfo;
    // What this build ran (the disclosure of the default flip):
    // the geometry model, the separation test and its parameter, the resolved
    // theta. Always filled at Create, legacy budget path or not.
    QfmmModelRecord _modelRecord;
};

QfmmJBuilder::QfmmJBuilder(std::shared_ptr<const State> state) : _state(std::move(state)) {}

std::size_t QfmmJBuilder::FarFieldPairCount() const noexcept {
    return _state->_farFieldPairs.size();
}

const std::optional<FockModeInfo>& QfmmJBuilder::ModeInfo() const noexcept {
    return _state->_modeInfo;
}

QfmmModelRecord QfmmJBuilder::ModelRecord() const noexcept {
    return _state->_modelRecord;
}

qcx::Result<QfmmJBuilder> QfmmJBuilder::Create(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
    const QfmmOptions& options) {
    if (options.lMult < -1 || options.lMult > internal::kQfmmMaxLMult)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "lMult must be -1 or in 0..kQfmmMaxLMult"});
    }

    if (options.maxLeafSize == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxLeafSize must be positive"});
    }

    // A NaN theta would silently classify everything as near field in the
    // IsWellSeparated comparisons (NaN >= x is always false) - reject it
    // instead of letting a broken option masquerade as the tight preset.
    if (std::isnan(options.theta))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "theta must not be NaN"});
    }

    // The point-group reduction, refused by name (the contract
    // RiFullFockBuilder::Create also names: honoured, refused with its condition named,
    // or demoted with the disclosure - never silently substituted).
    //
    // The refusal stands on the MECHANISM, not on the far field's size: this
    // builder does not implement the reduction at all, and the near field is
    // the nested DirectJkFockBuilder, whose own options are where a reduction
    // would have to be requested. It USED to also argue that there was little
    // to serve, quoting the retired midpoint/width criterion's far field - 36
    // of 122,319,961 CSR entries at C24H50/def2-SVP, a far field that was not
    // engaged. The 2026-09-15 default flip retired that model, and
    // under the surface-ball criterion the same fixture's far field carries
    // 37.72% of the CSR classes and 41.7% of the function quartets, so
    // "almost nothing to serve" is now false and is not claimed here.
    if (options.symmetryReduction != nullptr)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kUnimplemented,
            "QfmmJBuilder does not implement the point-group reduction, so a non-null "
            "symmetryReduction cannot be honoured: the near field that carries the work is the "
            "nested DirectJkFockBuilder, whose own options are where a reduction would have to "
            "be requested - clear the field for this builder. The far field this family would "
            "otherwise serve is NOT negligible under the corrected default (37.72% of the CSR "
            "classes, 41.7% of the function quartets at C24H50/def2-SVP), so this refusal denies "
            "a real reduction rather than an empty one"});
    }

    // The far field does carry one reuse that is NOT this reduction and does
    // not answer it: the M2L coefficient fold (one table evaluation per
    // (l, m) row, the pair's reversed direction served from the row parity -
    // ApplyM2LTo in internal/qfmm_multipole.cpp) shares a polynomial
    // evaluation between the two directions of a far pair. It assumes no
    // symmetry of the molecule, the octree or the density, reads no group and
    // takes no option, so it runs whether or not symmetryReduction is set -
    // it does not make the refusal above narrower, and the refusal does not
    // make it redundant.

    // The option resolution: theta < 0 forces the degenerate theta -> 0 gate
    // (nothing well-separated - everything near field, the bit-exact
    // acceptance gate), theta == 0 picks ThetaForPreset(accuracy), theta > 0
    // the explicit override; lMult == -1 picks LMultForPreset(accuracy), else
    // the explicit 0..kQfmmMaxLMult.
    double resolvedTheta = ThetaForPreset(options.accuracy);

    if (options.theta < 0.0)
    {
        resolvedTheta = 0.0;
    }

    if (options.theta > 0.0)
    {
        resolvedTheta = options.theta;
    }

    int resolvedLMult = LMultForPreset(options.accuracy);

    if (options.lMult != -1)
    {
        resolvedLMult = options.lMult;
    }

    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    // The adaptive-memory mode decision: the Create-time footprint estimate (the nested near-field
    // direct builder's estimate plus the outer store) against the budget's
    // remaining bytes, decided once. Only the options that carry a budget
    // take this path - the legacy path keeps its exact behavior below. The
    // near-field batch cap is the FockBuildOptions default (QfmmOptions has
    // no batch field); the near-field builder re-sweeps inside its own
    // Create (the same leaf-driven domain rows - no seam carries a prebuilt
    // pattern).
    std::optional<FockModeInfo> modeInfo;
    std::size_t nearFieldBatchBytes = FockBuildOptions{}.maxBatchBytes;
    // The budget path's counted sweep, carried out of the decision block to
    // the nested near-field's options (zero on the legacy path - no
    // carrier).
    std::size_t validatedSweepCount = 0;

    // The counted decision (the sweep, the estimate and the reservation)
    // sits below, after the tree and its near-field domain exist: the
    // estimate must count the same leaf-driven rows the nested near-field
    // builder enumerates at its own Create (the near-field domain is that carrier).
    // The a-priori exclusion (i) lives in that block for the same reason -
    // it bounds the NEAR-FIELD pattern, which exists only once the
    // interaction lists do (the pre-tree all-survive form it replaced
    // could not see the restriction at all and over-refused large-sparse
    // systems whose near-field rows are a small fraction of the pair
    // space).

    auto pairStore = internal::BuildPairData(molecule, basisSet, *pairList);

    if (!pairStore.has_value())
    {
        return std::unexpected(pairStore.error());
    }

    // The pair extent is preset-driven: the tail-charge argument couples the
    // geometric footprint to the accuracy budget - a looser preset admits a
    // larger missed-charge tail
    // at its own coarser budget and shrinks the boxes (more far-field
    // pairs), kTight keeps the committed 1e-10 cutoff.
    // The pair-geometry model: the recorded midpoint-padded bound by
    // default (bit-identical), the product-distribution ball when asked for.
    const internal::QfmmExtentModel extentModel =
        options.extentModel == QfmmExtentMode::kProductBall
            ? internal::QfmmExtentModel::kProductBall
            : internal::QfmmExtentModel::kMidpointBound;
    std::vector<internal::QfmmPairGeometry> geometries = internal::ComputePairGeometries(
        *pairStore, QfmmExtentForPreset(options.accuracy), extentModel);
    auto tree = internal::BuildQfmmTree(geometries, options.maxLeafSize);

    if (!tree.has_value())
    {
        return std::unexpected(tree.error());
    }

    std::vector<std::pair<std::size_t, std::size_t>> farFieldPairs;
    std::vector<std::pair<std::size_t, std::size_t>> nearFieldLeafPairs;
    internal::QfmmSeparation separation;
    separation.test = options.separationMode == QfmmSeparationMode::kSurfaceBall
                          ? internal::QfmmSeparationTest::kSurfaceBall
                          : internal::QfmmSeparationTest::kWidthTheta;
    separation.theta = resolvedTheta;
    separation.k = options.separationK;

    // The classification reads the preset's OWN angle: theta == 0 is "not
    // given" and resolves to ThetaForPreset(accuracy) above, which is what the
    // pair test is handed below. So every preset aims the far field at the
    // separation its own rung declares, and a written theta > 0 is the historic
    // explicit override - the same recorded contract, at the caller's number
    // instead of the rung's. theta < 0 is the degenerate gate (nothing well
    // separated, the far field empty, the bit-exact acceptance build) expressed
    // as k < 0 so it works under the surface form too.
    if (options.theta > 0.0)
    {
        separation.test = internal::QfmmSeparationTest::kWidthTheta;
    }

    if (options.theta < 0.0)
    {
        separation.k = -1.0;
    }

    // A preset whose own rung is the degenerate gate keeps it: kTight's
    // recorded identity is theta -> 0 (ThetaForPreset(kTight) = 0 and
    // LMultForPreset(kTight) = 0) - the exact near-field-only rung, its
    // separation criterion is the empty far field and nothing else. The
    // centre-to-width test already reads theta 0 as "nothing is well
    // separated", so this only pins the same gate under the surface form,
    // which would otherwise acquire a live far field at L = 0 - neither that
    // rung's contract nor an accuracy its budget was derived for. An explicit
    // theta still wins (theta > 0 selects the width test above).
    if (options.theta == 0.0 && ThetaForPreset(options.accuracy) <= 0.0)
    {
        separation.k = -1.0;
    }

    // The geometric pass: the separation exactly as written above, no budget.
    // Its far-pair count is the model's own number of interactions, and it is
    // the denominator the per-interaction share is split over.
    internal::BuildInteractionLists(*tree, separation, farFieldPairs, nearFieldLeafPairs);

    const std::size_t geometricFarPairCount = farFieldPairs.size();
    const double farFieldBudget = QfmmBudgetForPreset(options.accuracy);
    const double epsInt = geometricFarPairCount == 0
                              ? 0.0
                              : farFieldBudget / static_cast<double>(geometricFarPairCount);

    // The error arm: a pair is far only when some degree up to the resolved
    // order keeps the a-priori truncation bound within that interaction's share
    // of the preset budget; the pairs that fail descend into the near field,
    // where they are computed exactly. The arm lives INSIDE the separation
    // predicate the traversal classifies with, so the two lists still partition
    // the pair space - filtering the far list afterwards would drop pairs from
    // both fields. It sits after whichever geometric form classified the pair,
    // so it needs no form of its own.
    //
    // It is OFF by default, and that is the decided default rather than a
    // dangling switch: the bound it consults is a worst case over
    // distributions, so at the angle the presets declare it refuses pairs the
    // far field represents well inside budget - at the normal preset on the
    // reference fixture the arm refuses every pair the angle admits and empties
    // the far field, which is the opposite of what the preset ladder is for. A
    // run that wants admission certified rather than measured asks for it.
    const bool admissionErrorAware = options.enforceFarFieldBudget;

    if (admissionErrorAware)
    {
        separation.epsInt = epsInt;
        separation.orderCap = resolvedLMult;
        internal::BuildInteractionLists(*tree, separation, farFieldPairs, nearFieldLeafPairs);
    }

    const std::size_t pairsMovedToNearField = geometricFarPairCount - farFieldPairs.size();

    // The order selection:
    // per far-field pair the minimal multipole order whose a-priori
    // geometric bound (r / d)^(L+1) fits the preset budget split over the
    // interactions (epsInt = QfmmBudgetForPreset / nFar), capped at the
    // resolved order; useAdaptiveOrder == false keeps every pair at the
    // resolved order (the selector-free uniform run). No far pairs - no
    // selection; every node stays -1 below. The selector records its
    // fall-through: an interaction it could not bring within epsInt runs at
    // the cap with the budget missed, which the run record now says out loud
    // instead of leaving it to be inferred from an accuracy miss.
    std::vector<int> farPairOrders;
    internal::QfmmOrderSelectionStats orderSelection;

    if (!farFieldPairs.empty())
    {
        // The selection runs even on the uniform path, where its orders are not
        // used: the fall-through record is about the interactions the build is
        // about to run at the cap, and that is the same question whether the
        // order was chosen per pair or fixed for all of them. The pass is a
        // loop over the pair list - no integrals - so the record costs nothing
        // measurable.
        std::vector<int> selected = internal::SelectMultipoleOrders(
            *tree, farFieldPairs, epsInt, resolvedLMult, &orderSelection);

        farPairOrders = options.useAdaptiveOrder
                            ? std::move(selected)
                            : std::vector<int>(farFieldPairs.size(), resolvedLMult);
    }

    std::vector<int> nodeOrders =
        internal::ComputeNeededNodeOrders(*tree, farFieldPairs, farPairOrders);

    // The near-field domain:
    // the leaf-driven carrier of the near-field pair-pair set - for each
    // near-field leaf pair (A, B) the products of the two leaves' pair
    // indices (the diagonal A == A included, canonical ket <= bra) are the
    // pair-pairs of the set. The nested direct builder enumerates its
    // cached Schwarz neighbor list from these leaf pairs instead of
    // sweeping the full pair space and filtering every candidate through
    // the PairPairRestriction bitset (restrictToPairPairs stays for the
    // test harness - the qfmm_fixture.hpp gated sweep is the bit-identity
    // reference of that gate). Same candidate set under the same pair
    // cutoff, so the two near-field builds are bit-identical at the serial
    // pin; at theta -> 0 the domain covers every leaf pair, so the gate
    // build is the full list - the bit-exact direct path as before.
    LeafNearFieldDomain nearFieldDomain;
    nearFieldDomain.nLeaves = tree->nodes.size();
    nearFieldDomain.leafOfPair = tree->leafOfPair;
    nearFieldDomain.nearFieldLeafPairs = std::move(nearFieldLeafPairs);

    // The counted decision (the budget path's one and only half): the
    // near-field pattern count with its a-priori exclusion (i), then the
    // nested near-field builder's estimate on the leaf-driven rows plus
    // the outer store, against the budget's remaining bytes, decided
    // once.
    if (options.workspaceBudget != nullptr)
    {
        qcx::memory::WorkspaceBudget& budget = *options.workspaceBudget;
        const std::size_t remaining = budget.Remaining();
        FockModeInfo info;
        info.budgetBytes = budget.CapacityBytes();
        info.remainingAtDecision = remaining;
        info.maxBatchBytes = nearFieldBatchBytes;

        auto schwarz = ComputeSchwarzBounds(molecule, basisSet);

        if (!schwarz.has_value())
        {
            return std::unexpected(schwarz.error());
        }

        // The exact near-field pattern count: the leaf-driven
        // BuildNeighborList pass-1 candidate count without building the CSR
        // (the exclusion now reads this counted form). The
        // count is bit-identical to the build's own counting pass - the
        // same per-leaf lists and the same product cutoff over the same
        // near-field leaf pairs - and to the rows the nested near-field
        // builder enumerates at its own Create, so the outer's model is
        // exactly what the nested reserves: a full-sweep count would
        // charge far-field pattern bytes (8 per row) no builder allocates.
        const std::size_t patternCount =
            internal::CountNearFieldPatternEntries(*schwarz, options.accuracy, nearFieldDomain);

        // A-priori exclusion (i): even the counted near-field pattern
        // cannot fit - no fast path, LightPath mandatory (the all-survive
        // form that refused before any tree existed is gone: it could not
        // see the near-field restriction, so it over-refused large-sparse
        // systems - the all-survive bound measured 24.44 GiB against a
        // ~38 MB counted pattern - and the exact count now runs wherever
        // the decision runs). The x8 form is integer-exact: count >
        // remaining / 8 is 8 x count > remaining without the size overflow.
        if (patternCount > remaining / 8)
        {
            info.patternExcluded = true;
            info.mode = FockBuildMode::kLightPath;
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           internal::FastPathRefusal("QFMM", 0, remaining, true, false)});
        }

        // The counted pattern is the outer's validation of the stack; the
        // nested near-field skips its a-priori exclusion (i) when non-zero
        // (an asymmetry between the two stacks: the RI-J's nested exchange
        // already skipped it when the outer supplied a counted pattern, the
        // QFMM's did not).
        validatedSweepCount = patternCount;

        const std::size_t threadCount =
            static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
        // The near-field estimate: the nested builder is Coulomb-only
        // (buildCoulombOnly below, the J-only QFMM stack) - no exchange
        // pass - so the exchange term would be phantom here; the
        // exchangeEngaged flag zeroes it.
        internal::DirectFootprintTerms nearField = internal::DirectFootprint(molecule,
                                                                             basisSet,
                                                                             *pairList,
                                                                             patternCount,
                                                                             nearFieldBatchBytes,
                                                                             threadCount,
                                                                             false,
                                                                             0,
                                                                             options.accuracy,
                                                                             false);
        const std::size_t outerStore =
            internal::QfmmOuterStoreBytes(molecule, basisSet, *pairList, options.maxLeafSize);
        const std::size_t predicted = nearField.Total() + outerStore;

        // Clamp first: shrink the per-thread batch arena of the near-field
        // builder by the deficit; only a batch that still cannot fit at a
        // unit arena descends the ladder.
        const std::size_t clamped =
            internal::ClampBatchBytes(nearFieldBatchBytes, threadCount, predicted, remaining);

        if (clamped == 0)
        {
            info.mode = FockBuildMode::kLightPath;
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           internal::FastPathRefusal("QFMM", predicted, remaining, false, false)});
        }

        // Re-estimate with the clamped batch - the fired terms (same
        // Coulomb-only gating as above).
        nearField = internal::DirectFootprint(molecule,
                                              basisSet,
                                              *pairList,
                                              patternCount,
                                              clamped,
                                              threadCount,
                                              false,
                                              0,
                                              options.accuracy,
                                              false);
        info.predictedBytes = nearField.Total() + outerStore;
        info.pairStoreBytes = nearField.pairStoreBytes;
        info.patternBytes = nearField.patternBytes;
        info.scratchBytes = nearField.scratchBytes;
        info.structuralBytes = nearField.structuralBytes;
        info.outerStoreBytes = outerStore;
        info.maxBatchBytes = clamped;
        info.mode = FockBuildMode::kFastPath;

        // The outer reservation: the full estimate minus the nested
        // near-field builder's own estimate - exactly the outer store. The
        // nested near-field charges its own parts at its own Create (the
        // same budget pointer, the same clamped batch - nesting order =
        // reservation order), so the total stack charge is exactly
        // predictedBytes and the nested fits whenever the outer does.
        const std::size_t outerReservation = info.predictedBytes - nearField.Total();

        if (!budget.Reserve(outerReservation))
        {
            // A concurrent reservation consumed the budget between the
            // decision and the reserve (the CWA is monotone - the failed
            // reserve charges nothing) - the decision is stale, refuse.
            info.mode = FockBuildMode::kLightPath;
            info.reservedBytes = 0;
            return std::unexpected(qcx::Error{
                qcx::ErrorCode::kUnimplemented,
                internal::FastPathRefusal("QFMM", info.predictedBytes, remaining, false, false) +
                    " (the budget moved between the decision and the reservation)"});
        }

        info.reservedBytes = outerReservation;
        nearFieldBatchBytes = clamped;
        modeInfo = info;
    }

    FockBuildOptions nearFieldOptions;
    nearFieldOptions.accuracy = options.accuracy;
    nearFieldOptions.buildCoulombOnly = true;
    nearFieldOptions.useDensityScreening = options.useDensityScreening;
    nearFieldOptions.useCertifiedMixedPrecision = options.useCertifiedMixedPrecision;
    nearFieldOptions.maxParallelChunks = options.maxParallelChunks;
    nearFieldOptions.maxBatchBytes = nearFieldBatchBytes;
    // The outer's counted sweep (the decision block above): the nested
    // skips its a-priori all-survive exclusion (i) - the all-survive bound
    // can exceed the post-reservation remaining even when the counted
    // pattern fits (the stack the outer proved feasible).
    nearFieldOptions.validatedSweepCount = validatedSweepCount;
    // The SAME budget pointer: the nested near-field charges its own parts
    // at its own Create (the outer reserved the difference - nesting order
    // = reservation order).
    nearFieldOptions.workspaceBudget = options.workspaceBudget;
    nearFieldOptions.leafNearFieldDomain = std::move(nearFieldDomain);
    auto nearFieldBuilder =
        DirectJkFockBuilder::Create(molecule, basisSet, coreHamiltonian, nearFieldOptions);

    if (!nearFieldBuilder.has_value())
    {
        return std::unexpected(nearFieldBuilder.error());
    }

    // The per-pair moment orders: a canonical pair lives in exactly one
    // leaf (its expansion center sits in one box), and its needed order is
    // that leaf's band - -1 for a pair in a near-field-only leaf, which the
    // moment table then allocates nothing (the on-demand table).
    std::vector<int> pairOrders(pairStore->size(), -1);

    for (std::size_t p = 0; p < pairStore->size(); ++p)
    {
        pairOrders[p] = nodeOrders[tree->leafOfPair[p]];
    }

    auto momentTable = internal::BuildMomentTable(*pairList, *pairStore, geometries, pairOrders);

    if (!momentTable.has_value())
    {
        return std::unexpected(momentTable.error());
    }

    auto state = std::make_shared<State>();
    state->_pairList = std::move(*pairList);
    state->_pairStore = std::move(*pairStore);
    state->_geometries = std::move(geometries);
    state->_tree = std::move(*tree);
    state->_farFieldPairs = std::move(farFieldPairs);
    state->_farPairOrders = std::move(farPairOrders);
    state->_nodeOrders = std::move(nodeOrders);
    state->_momentTable = std::move(*momentTable);
    state->_maxParallelChunks = options.maxParallelChunks;
    state->_nearFieldBuilder.emplace(std::move(*nearFieldBuilder));
    state->_modeInfo = modeInfo;
    // The record states what RAN, not what was asked for: a positive theta
    // resolves the width form at that override, a negative theta and every
    // k < 0 form resolve the gate, and the absent theta resolves the width form
    // at the preset's own angle - so theta here is nonzero even when the caller
    // wrote nothing, which is the point of the field.
    state->_modelRecord.extentModel = options.extentModel;
    state->_modelRecord.separationMode =
        separation.test == internal::QfmmSeparationTest::kSurfaceBall
            ? QfmmSeparationMode::kSurfaceBall
            : QfmmSeparationMode::kWidthTheta;
    state->_modelRecord.separationK =
        separation.test == internal::QfmmSeparationTest::kSurfaceBall ? separation.k : 0.0;
    state->_modelRecord.theta =
        separation.test == internal::QfmmSeparationTest::kWidthTheta ? resolvedTheta : 0.0;

    // The budget record: what the far field was held to, what it cost in
    // pairs, and whether any interaction ran at the cap with the budget
    // missed. The last field is the one that was missing - a build whose far
    // field overspent its preset budget used to look exactly like one that
    // met it.
    state->_modelRecord.errorAwareAdmission = admissionErrorAware;
    state->_modelRecord.farFieldBudget = farFieldBudget;
    state->_modelRecord.perInteractionBudget = epsInt;
    state->_modelRecord.geometricFarPairCount = geometricFarPairCount;
    state->_modelRecord.pairsMovedToNearField = pairsMovedToNearField;
    state->_modelRecord.fellThroughToCap = orderSelection.fellThroughToCap;
    state->_modelRecord.worstTruncationBound = orderSelection.worstBound;

    return QfmmJBuilder(std::move(state));
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> QfmmJBuilder::BuildFock(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density) const {
    const State& state = *_state;
    const std::size_t n = state._pairList.functionCount;

    if (density.Shape()[0] != n || density.Shape()[1] != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "density shape mismatch"});
    }

    // The near field: H + 2J_near(rho) from the restricted direct builder.
    auto nearField = state._nearFieldBuilder->BuildFock(density);

    if (!nearField.has_value())
    {
        return std::unexpected(nearField.error());
    }

    Eigen::MatrixXd fock = internal::TensorToEigen(*nearField);

    // The far field: the density-weighted leaf moments, the M2M pass, the
    // M2L + L2L potentials, and the 2J accumulation in the direct builder's
    // convention (AccumulateBlock in fock_build.cpp) - the composed result
    // is H + 2J_near + 2J_far = H + 2J.
    Eigen::MatrixXd densityMatrix = internal::TensorToEigen(density);

    // The chunk override rides every far-field pass (the near field
    // already carries it in its own FockBuildOptions): each pass resolves
    // the 0/1/>=2 value on its own task count. AggregateNodeMoments stays
    // serial - its reverse pre-order children-before-parents dependency
    // does not decompose into the single-writer subtrees of the other
    // passes, so it stays out of the chunked decomposition (the M2M is a small
    // fraction of the far field's work).
    const auto leafMoments = internal::AggregateLeafMoments(state._pairList,
                                                            state._tree,
                                                            state._momentTable,
                                                            state._geometries,
                                                            densityMatrix,
                                                            state._nodeOrders,
                                                            state._maxParallelChunks);
    const auto nodeMoments =
        internal::AggregateNodeMoments(state._tree, leafMoments, state._nodeOrders);
    const auto potentials = internal::BuildFarFieldPotentials(state._tree,
                                                              state._farFieldPairs,
                                                              nodeMoments,
                                                              state._farPairOrders,
                                                              state._nodeOrders,
                                                              state._maxParallelChunks);
    internal::AccumulateFarFieldJ(fock,
                                  state._pairList,
                                  state._tree,
                                  state._momentTable,
                                  potentials,
                                  state._geometries,
                                  state._nodeOrders,
                                  state._maxParallelChunks);

    return internal::EigenToTensor(fock);
}

} // namespace qcx::integrals
