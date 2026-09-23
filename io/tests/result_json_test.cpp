// Result-JSON serialization acceptance: the schema_version constant, the
// null-vs-fabricated policy of the optional members, and the
// resources_resolved block — the caps, the cap_note honesty policy, the ri_j
// budget-path records (workspace_budget + mode_record),
// the selection record, the full-group
// labeling blocks (schema 8), the certified-bound block (schema
// 14) with its absent-vs-true-zero distinction, the compute-profile
// block (schema 20) with its measured-vs-unmeasured distinction, and the
// ri_chunk_bytes request record (schema 23) with its honoured-vs-dropped
// distinction, and the symmetry-blocking disclosure (schema 32) with
// its used-vs-demoted distinction and its absence on a run that never
// requested the blocking. Schema 34's payload is the run's own physics: the
// always-present `method` word, and the Kohn-Sham-only `functional` and
// `xc_grid` pair, whose ABSENCE on a Hartree-Fock record is the statement.
#include "qcx/io/result_json.hpp"

#include <gtest/gtest.h>
#include <string>
#include <utility>

namespace {

// The key-count helper: nlohmann's dump(2) pretty-prints arrays multi-line,
// so a formatted-array pin would be format-fragile; counting the key
// occurrences pins the record structure instead.
int CountOccurrences(const std::string& text, const std::string& needle) {
    int count = 0;

    for (std::size_t pos = text.find(needle); pos != std::string::npos;
         pos = text.find(needle, pos + needle.size()))
    {
        ++count;
    }

    return count;
}

TEST(ResultJsonTest, SchemaVersionIsForty) {
    // v34 -> v35: the run record says what it RAN in the orthogonal builder-axis
    // vocabulary, one bump for one payload - the top-level `builder_axes` block
    // (RunBuilderAxes): `integral_family`, `storage_tier`, `execution_backend`,
    // `requested_by`, and `legacy_spelling` with its `deprecated` note exactly
    // when the input named its selection through the deprecated
    // `[method] fock_builder` key.
    // Keys added, so a consumer must notice - the documented bump rule.
    //
    // Why the block exists, in one sentence: the selection record names the
    // builder in the LEGACY word, where a lean run reports "direct", so
    // before this key a reader could not see the TIER at all without
    // already knowing which words were tier words.
    //
    // No existing key moved, or was removed, and no nullability rule changed:
    // `builder_axes` is always present (every run resolves a builder, so the
    // triple is always determinable) while `legacy_spelling` inside it is the
    // absent half, and `requested_by` is what keeps that absence unambiguous.
    //
    // The rename of this test is part of the pin: a version pin that kept its
    // old name while asserting the new number would be the drift this suite
    // exists to catch.
    //
    // v35 -> v36: the run record LOSES `resources_resolved.memory_model` and
    // every key under it (the driver's hand-derived commit-peak prediction,
    // deleted rather than repaired). A removed
    // key is a change a consumer must notice, so the number moves with it. The
    // memory limit is still enforced - the job-object cap the driver applies
    // (`process_caps`) is untouched by the deletion - and the rung a run takes
    // is still decided, by the integrals engine's own Create-time estimates.
    // The one fact the removed block carried that no other key carries is the
    // within-family member of the direct family, and `selection.builder_member`
    // (schema 15) is where that member is named.
    //
    // v36 -> v37: the `qfmm_model` block GAINS the far field's accuracy
    // budget - the error arm's own flag, the preset budget the field was held
    // to, the share each interaction was allowed, the geometric far-pair
    // count it was split over, the pairs the budget moved into the near
    // field, the interactions that ran at the order cap with their bound
    // above that share, and the worst such bound. The engine computed every
    // one of them already and the builder's own record carried them; only the
    // serializer dropped them, so a run whose far field missed its budget
    // used to be indistinguishable in the document from one that met it.
    // Keys added, so the number moves with them.
    //
    // v37 -> v38: NO key is added and none is removed - what moves is the
    // PRESENCE RULE of `resources_resolved.eri_store` (RunEriStore, schema
    // 33), widened to the unrestricted legs. At v33 an unrestricted (UHF/UKS)
    // run that named `method.eri_cache_store` was refused by name before it
    // could serialize, so the block was a restricted-run-only shape; the
    // engine-decorator seam is now wired on the direct family's per-spin
    // coulomb and exchange halves, so those runs carry the same
    // honoured-or-demoted block their restricted sibling does. A consumer
    // keyed on "no `eri_store` on an unrestricted run" is the one that must
    // notice - the same reasoning v25 used when it widened
    // `selection.approximation` to exact-kernel runs carrying a notice.
    //
    // v38 -> v39: the `builder_axes` block GAINS `device` (RunBuilderAxes), the
    // device the run REQUIRED in the input's own selector vocabulary ("host" |
    // "cuda:<index>"), present exactly when the file wrote `[builder] device`.
    // A key added, so the number moves with it. It carries no requested-vs-ran
    // pairing and needs none: a requirement the resolved builder cannot supply
    // is refused by name before serialization, so a document carrying this key
    // is one whose kernels executed where the input said they must.
    // v39 -> v40: the `properties` block GAINS `xc_gradient` (RunXcGradient),
    // the fixed-density exchange-correlation contribution to the nuclear
    // gradient: the 3N vector in hartree/bohr and the energy the same walk
    // integrated, absent unless the run asked for it. A key added, so the
    // number moves with it - and it is the first record member that is a
    // DERIVATIVE of an energy the same document carries, which is why the
    // block's own name states the fixed-density limitation rather than leaving
    // a reader to read the vector as a total nuclear gradient.
    EXPECT_EQ(qcx::io::RunResult::kSchemaVersion, 40);
}

TEST(ResultJsonTest, MethodIsAlwaysWrittenAndNamesTheRunPath) {
    // The record names the physics it was produced by. `method` is
    // always present - every run is one of the four words - and the spelling
    // is the input's own vocabulary, so a consumer reads "rks" here and can
    // write "rks" into a file.
    const std::pair<qcx::io::MethodType, const char*> words[] = {
        {qcx::io::MethodType::kRhf, "rhf"},
        {qcx::io::MethodType::kUhf, "uhf"},
        {qcx::io::MethodType::kRks, "rks"},
        {qcx::io::MethodType::kUks, "uks"},
    };

    for (const auto& [type, word] : words)
    {
        qcx::io::RunResult result;
        result.method = type;

        const std::string json = qcx::io::SerializeRunResultJson(result);
        EXPECT_NE(json.find(std::string("\"method\": \"") + word + "\""), std::string::npos)
            << "word: " << word;
    }

    // The struct's own default is the input's default (kRhf), so a result no
    // driver filled still names a method rather than omitting the key - and
    // the serializer never writes a null here.
    qcx::io::RunResult result;
    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"method\": \"rhf\""), std::string::npos);
    EXPECT_EQ(json.find("\"method\": null"), std::string::npos);
}

TEST(ResultJsonTest, FunctionalAndXcGridAreTheKohnShamOnlyPayloads) {
    // The functional and the grid are written exactly when the run
    // had them, and their absence is the statement for a run that did not.
    // A Hartree-Fock record must carry NEITHER key - not a null and not an
    // xc_grid of defaults, which would be a claim about a functional nobody
    // named and a quadrature that never touched those numbers.
    qcx::io::RunResult hf;
    hf.method = qcx::io::MethodType::kRhf;
    const std::string hfJson = qcx::io::SerializeRunResultJson(hf);
    EXPECT_EQ(hfJson.find("\"functional\""), std::string::npos);
    EXPECT_EQ(hfJson.find("\"xc_grid\""), std::string::npos);

    // The same for an unrestricted Hartree-Fock run: the absence follows the
    // PATH (no density functional, no XC grid), not the spin.
    qcx::io::RunResult uhf;
    uhf.method = qcx::io::MethodType::kUhf;
    const std::string uhfJson = qcx::io::SerializeRunResultJson(uhf);
    EXPECT_EQ(uhfJson.find("\"functional\""), std::string::npos);
    EXPECT_EQ(uhfJson.find("\"xc_grid\""), std::string::npos);

    // A Kohn-Sham run carries both: the RESOLVED functional name and the six
    // resolved grid settings, one key each, spelled as the input spells them.
    qcx::io::RunResult rks;
    rks.method = qcx::io::MethodType::kRks;
    rks.functional = "slater";
    rks.xcGrid = qcx::io::RunXcGrid{40, 50, 0.6, 3, 1e-12, 64};
    const std::string rksJson = qcx::io::SerializeRunResultJson(rks);
    EXPECT_NE(rksJson.find("\"method\": \"rks\""), std::string::npos);
    EXPECT_NE(rksJson.find("\"functional\": \"slater\""), std::string::npos);
    EXPECT_NE(rksJson.find("\"radial_points\": 40"), std::string::npos);
    EXPECT_NE(rksJson.find("\"angular_points\": 50"), std::string::npos);
    EXPECT_NE(rksJson.find("\"alpha\": 0.6"), std::string::npos);
    EXPECT_NE(rksJson.find("\"radial_exponent\": 3"), std::string::npos);
    EXPECT_NE(rksJson.find("\"trim_weight\": 1e-12"), std::string::npos);
    EXPECT_NE(rksJson.find("\"block_target\": 64"), std::string::npos);

    // The six keys are one block, not six top-level members: `xc_grid` is the
    // only place they appear, so a consumer cannot read a grid setting off a
    // record that built no grid.
    EXPECT_EQ(CountOccurrences(rksJson, "\"radial_points\""), 1);
    EXPECT_EQ(CountOccurrences(rksJson, "\"xc_grid\""), 1);
}

TEST(ResultJsonTest, MoleculeUnitsDisclosesTheInterpretedUnit) {
    // The record is self-describing about the one convention that has bitten
    // this project twice. Both words are published as such, and the default
    // reads "angstrom" rather than being absent - an unset member would put the
    // ambiguity back, and the parser's default IS Angstrom (a default is
    // the system's judgement, and a judgement can be stated).
    qcx::io::RunResult result;
    result.converged = true;
    EXPECT_NE(qcx::io::SerializeRunResultJson(result).find("\"molecule_units\": \"angstrom\""),
              std::string::npos);

    result.moleculeUnits = qcx::io::CoordinateUnit::kBohr;
    EXPECT_NE(qcx::io::SerializeRunResultJson(result).find("\"molecule_units\": \"bohr\""),
              std::string::npos);

    result.moleculeUnits = qcx::io::CoordinateUnit::kAngstrom;
    EXPECT_NE(qcx::io::SerializeRunResultJson(result).find("\"molecule_units\": \"angstrom\""),
              std::string::npos);
}

TEST(ResultJsonTest, RemovedOverlapDirectionsAreDisclosed) {
    // The linear-dependence removal's disclosure (schema 27): a run that
    // removed directions must SAY so, because a silent truncation is
    // indistinguishable from a correct run. The count is a number when a loop
    // filled it - 0 included - and null when no loop ran.
    qcx::io::RunResult result;
    result.converged = true;
    result.numRemovedOverlapDirections = 1u;

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"num_removed_overlap_directions\": 1"), std::string::npos);

    // Zero is a MEASUREMENT (a well-conditioned system), and it is published
    // as one: the well-conditioned case must not be confused with the
    // unobserved one.
    qcx::io::RunResult clean;
    clean.converged = true;
    clean.numRemovedOverlapDirections = 0u;
    EXPECT_NE(qcx::io::SerializeRunResultJson(clean).find("\"num_removed_overlap_directions\": 0"),
              std::string::npos);
}

TEST(ResultJsonTest, RemovedOverlapDirectionsAreNullWhenUnobserved) {
    // The same null rule as the achieved residuals: a result no SCF loop
    // filled publishes null, never a fabricated 0 - which here would also be
    // the value that means "well-conditioned".
    qcx::io::RunResult result;
    result.converged = false;

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"num_removed_overlap_directions\": null"), std::string::npos);
}

TEST(ResultJsonTest, AchievedResidualsQualifyConverged) {
    // The pair is what makes a bare `converged` readable: the gate's own
    // operands, recorded. Present as JSON numbers when an SCF loop filled
    // them.
    qcx::io::RunResult result;
    result.converged = true;
    result.iterations = 9;
    result.energyDeltaHartree = 2.2909808222948413e-05;
    result.rmsDensityDelta = 5.7899863807497482e-07;

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"converged\": true"), std::string::npos);
    EXPECT_NE(json.find("\"energy_delta_hartree\": 2.2909808222948413e-05"), std::string::npos);
    EXPECT_NE(json.find("\"rms_density_delta\": 5.789986380749748e-07"), std::string::npos);
}

TEST(ResultJsonTest, AchievedResidualsAreNullWhenUnobserved) {
    // Absent-vs-zero (the certified_bound discipline): a result no SCF loop
    // filled must not publish 0.0, which would read as perfect convergence.
    qcx::io::RunResult result;
    result.converged = false;

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"energy_delta_hartree\": null"), std::string::npos);
    EXPECT_NE(json.find("\"rms_density_delta\": null"), std::string::npos);
}

TEST(ResultJsonTest, TermCountersBlockEmittedWhenPresent) {
    // The term_counters block of an instrumented ri_j run: exactly the five
    // engine-emitted ids as non-negative integers (the record contract).
    qcx::io::RunResult result;
    result.termCounters = qcx::io::RunTermCounters{44, 891, 32356, 6014, 118429};

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"term_counters\""), std::string::npos);
    EXPECT_NE(json.find("\"x\": 44"), std::string::npos);
    EXPECT_NE(json.find("\"p3\": 891"), std::string::npos);
    EXPECT_NE(json.find("\"g3\": 32356"), std::string::npos);
    EXPECT_NE(json.find("\"qx\": 6014"), std::string::npos);
    EXPECT_NE(json.find("\"gx\": 118429"), std::string::npos);
}

TEST(ResultJsonTest, TermCountersBlockAbsentWhenUnset) {
    // The honesty policy: no term_counters key on an uninstrumented run -
    // absent, never a fabricated zero block.
    qcx::io::RunResult result;
    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_EQ(json.find("term_counters"), std::string::npos);
}

TEST(ResultJsonTest, CertifiedBoundBlockEmittedWhenPresent) {
    // The certified-bound block of a machinery run: the delivered
    // density-weighted kernel-bound sum in hartree, with the sample count
    // it was reduced over.
    qcx::io::RunResult result;
    result.certifiedBound = qcx::io::RunCertifiedBound{20, 1.25e-3, 3.5e-3};

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"certified_bound\""), std::string::npos);
    EXPECT_NE(json.find("\"calls\": 20"), std::string::npos);
    EXPECT_NE(json.find("\"last_call_ha\""), std::string::npos);
    EXPECT_NE(json.find("\"max_call_ha\""), std::string::npos);
}

TEST(ResultJsonTest, CertifiedBoundEnforcementMembersEmitted) {
    // Schema 16: the enforcement's own reading, on the same block as the
    // delivered bound. The two quantities are different scales and must
    // never be conflated - the routed sum is the one compared against the
    // budget, the delivered sum is what the lane's arithmetic cost.
    qcx::io::RunResult result;
    result.certifiedBound =
        qcx::io::RunCertifiedBound{20, 0.0, 5.96, true, 7.69e-11, 7.51e-5, 0, true};

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"enforced\": true"), std::string::npos);
    EXPECT_NE(json.find("\"budget_ha\""), std::string::npos);
    EXPECT_NE(json.find("\"routed_ha\""), std::string::npos);
    EXPECT_NE(json.find("\"routed_quartets\": 0"), std::string::npos);
    EXPECT_NE(json.find("\"fell_back_to_fp64\": true"), std::string::npos);
}

TEST(ResultJsonTest, CertifiedBoundEnforcementMembersDefaultToOff) {
    // The three-argument block a pre-schema-16 producer (and every
    // unenforced run) carries: the enforcement members are still emitted,
    // false/zero, so a consumer reads the same shape and knows the check
    // did not run. An absent key there would force the reader to infer.
    qcx::io::RunResult result;
    result.certifiedBound = qcx::io::RunCertifiedBound{20, 1.25e-3, 3.5e-3};

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"enforced\": false"), std::string::npos);
    EXPECT_NE(json.find("\"budget_ha\": 0.0"), std::string::npos);
    EXPECT_NE(json.find("\"routed_ha\": 0.0"), std::string::npos);
    EXPECT_NE(json.find("\"routed_quartets\": 0"), std::string::npos);
    EXPECT_NE(json.find("\"fell_back_to_fp64\": false"), std::string::npos);
}

TEST(ResultJsonTest, CertifiedBoundBlockAbsentWhenUnset) {
    // The honesty policy: the lean member's Schwarz-only builder computes
    // no bound at all, and the families without the out-parameter compute
    // none either - so the key is ABSENT rather than a fabricated 0.0
    // that would read as "certified with zero error".
    qcx::io::RunResult result;
    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_EQ(json.find("certified_bound"), std::string::npos);
}

TEST(ResultJsonTest, CertifiedBoundZeroIsATrueZeroNotAbsence) {
    // A machinery run whose lane delivered nothing (kTight's gate is 0.0)
    // still OBSERVED the quantity: the block is present carrying the true
    // zero, distinguishable from the unset case above by its presence.
    qcx::io::RunResult result;
    result.certifiedBound = qcx::io::RunCertifiedBound{20, 0.0, 0.0};

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"certified_bound\""), std::string::npos);
    EXPECT_NE(json.find("\"calls\": 20"), std::string::npos);
    EXPECT_NE(json.find("\"last_call_ha\": 0.0"), std::string::npos);
    EXPECT_NE(json.find("\"max_call_ha\": 0.0"), std::string::npos);
}

TEST(ResultJsonTest, QfmmModelBlockIsEmittedWhenPresent) {
    // Schema 29: a QFMM run's record carries the model its Coulomb half ran,
    // and the two member pairs a reader needs are both written - the extent
    // model and the separation test - with the separation buffer and the
    // resolved theta beside them. The values here are the engine's DEFAULT
    // pair; the writer copies them from the engine, so this test
    // pins the SERIALIZER, not the default.
    qcx::io::RunResult result;
    result.converged = true;
    qcx::io::RunQfmmModel model;
    model.extentModel = "kProductBall";
    model.separationMode = "kSurfaceBall";
    model.separationK = 1.0;
    model.theta = 0.3;
    result.qfmmModel = model;

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"qfmm_model\""), std::string::npos);
    EXPECT_NE(json.find("\"extent_model\": \"kProductBall\""), std::string::npos);
    EXPECT_NE(json.find("\"separation_mode\": \"kSurfaceBall\""), std::string::npos);
    EXPECT_NE(json.find("\"separation_k\": 1.0"), std::string::npos);
    EXPECT_NE(json.find("\"theta\": 0.3"), std::string::npos);

    // The retired pair is a different record, not a different block: the same
    // four members carry it, so a consumer that reads one reads both. This is
    // the leg that fails if the words are ever folded into a boolean.
    result.qfmmModel->extentModel = "kMidpointBound";
    result.qfmmModel->separationMode = "kWidthTheta";
    const std::string retired = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(retired.find("\"extent_model\": \"kMidpointBound\""), std::string::npos);
    EXPECT_NE(retired.find("\"separation_mode\": \"kWidthTheta\""), std::string::npos);
}

TEST(ResultJsonTest, QfmmModelBlockCarriesTheFarFieldBudget) {
    // Schema 37: the block carries the far field's accuracy budget as well as
    // the model, so a run whose far field missed the budget it was held to
    // states the miss rather than leaving a consumer to infer it. This pins
    // the SERIALIZER's key set and the numbers that ride it - the engine's own
    // values, which the writer copies verbatim.
    qcx::io::RunResult result;
    result.converged = true;
    qcx::io::RunQfmmModel model;
    model.errorAwareAdmission = true;
    model.farFieldBudget = 0.5;
    model.perInteractionBudget = 0.25;
    model.geometricFarPairCount = 400;
    model.pairsMovedToNearField = 12;
    model.fellThroughToCap = 3;
    model.worstTruncationBound = 0.125;
    result.qfmmModel = model;

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"error_aware_admission\": true"), std::string::npos);
    EXPECT_NE(json.find("\"far_field_budget\": 0.5"), std::string::npos);
    EXPECT_NE(json.find("\"per_interaction_budget\": 0.25"), std::string::npos);
    EXPECT_NE(json.find("\"geometric_far_pair_count\": 400"), std::string::npos);
    EXPECT_NE(json.find("\"pairs_moved_to_near_field\": 12"), std::string::npos);
    EXPECT_NE(json.find("\"fell_through_to_cap\": 3"), std::string::npos);
    EXPECT_NE(json.find("\"worst_truncation_bound\": 0.125"), std::string::npos);

    // The members are written even at their defaults, with the arm's own flag
    // as the answer: a build that did not arm the error arm still reports the
    // bound's verdict on it, so a consumer never has to read an absent key as
    // "the budget was met" (the `certified_bound` rule).
    qcx::io::RunQfmmModel unarmed;
    unarmed.extentModel = "kProductBall";
    unarmed.separationMode = "kSurfaceBall";
    unarmed.separationK = 1.0;
    unarmed.farFieldBudget = 0.5;
    unarmed.geometricFarPairCount = 400;
    result.qfmmModel = unarmed;
    const std::string defaults = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(defaults.find("\"error_aware_admission\": false"), std::string::npos);
    EXPECT_NE(defaults.find("\"far_field_budget\": 0.5"), std::string::npos);
    EXPECT_NE(defaults.find("\"per_interaction_budget\": 0.0"), std::string::npos);
    EXPECT_NE(defaults.find("\"geometric_far_pair_count\": 400"), std::string::npos);
    EXPECT_NE(defaults.find("\"pairs_moved_to_near_field\": 0"), std::string::npos);
    EXPECT_NE(defaults.find("\"fell_through_to_cap\": 0"), std::string::npos);
    EXPECT_NE(defaults.find("\"worst_truncation_bound\": 0.0"), std::string::npos);
}

TEST(ResultJsonTest, QfmmModelBlockAbsentWhenNoQfmmBuildRan) {
    // The absence rule, not a null: a run that wired no composed-QFMM builder
    // has no model to report, and a fabricated default block would claim a
    // build ran over objects it never touched - the reason `certified_bound`
    // and `term_counters` are absent rather than zeroed on the paths that
    // cannot back them.
    const qcx::io::RunResult result;
    EXPECT_EQ(qcx::io::SerializeRunResultJson(result).find("\"qfmm_model\""), std::string::npos);
    EXPECT_EQ(qcx::io::SerializeRunResultJson(result).find("\"extent_model\""), std::string::npos);
}

TEST(ResultJsonTest, MoldenBlockEmittedWhenRequested) {
    qcx::io::RunResult result;
    qcx::io::RunProperties properties;
    properties.molden = qcx::io::RunMolden{"C:/runs/h2.molden"};
    result.properties = std::move(properties);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"molden\""), std::string::npos);
    EXPECT_NE(json.find("\"file\": \"C:/runs/h2.molden\""), std::string::npos);
}

TEST(ResultJsonTest, MoldenBlockAbsentWhenNotRequested) {
    // The honesty policy: no molden key, no molden member - absent, never
    // a placeholder.
    qcx::io::RunResult result;
    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_EQ(json.find("molden"), std::string::npos);
}

TEST(ResultJsonTest, XcGradientBlockEmittedWhenRequested) {
    // The fixed-density exchange-correlation contribution: the 3N vector and
    // the energy the same walk integrated. dump(2) writes one array element
    // per line, so the vector is pinned by its key and by each element's own
    // text rather than by a whole-array line the format would decide; the six
    // values below are distinct as text, so each count pins one element.
    qcx::io::RunResult result;
    qcx::io::RunProperties properties;
    properties.xcGradient =
        qcx::io::RunXcGradient{{0.125, -0.375, 0.625, -0.875, 1.125, -1.375}, -8.125};
    result.properties = std::move(properties);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_EQ(CountOccurrences(json, "\"xc_gradient\""), 1);
    EXPECT_EQ(CountOccurrences(json, "\"gradient\": ["), 1);

    for (const char* element : {"0.125", "-0.375", "0.625", "-0.875", "1.125", "-1.375"})
    {
        EXPECT_EQ(CountOccurrences(json, element), 1) << element;
    }

    EXPECT_EQ(CountOccurrences(json, "\"energy_hartree\": -8.125"), 1);
}

TEST(ResultJsonTest, XcGradientBlockAbsentWhenNotRequested) {
    // Absent, never a zero-filled block: a record without the key states that
    // no walk ran, which a block of zeros would misstate as a zero
    // contribution.
    qcx::io::RunResult result;
    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_EQ(json.find("xc_gradient"), std::string::npos);
}

TEST(ResultJsonTest, SelectionBlockEmittedWhenPresent) {
    // The selection record: builder + reasoning always, picked and
    // explicit_builder only when they apply.
    qcx::io::RunResult result;
    qcx::io::RunSelection selection;
    selection.builder = qcx::io::BuilderKind::kGpu;
    selection.picked = qcx::io::BuilderKind::kGpu;
    selection.reasoning = "gpu fastest at n=562 on 4 GiB";
    result.resourcesResolved.selection = std::move(selection);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"selection\""), std::string::npos);
    EXPECT_NE(json.find("\"builder\": \"gpu\""), std::string::npos);
    EXPECT_NE(json.find("\"picked\": \"gpu\""), std::string::npos);
    EXPECT_NE(json.find("\"reasoning\": \"gpu fastest at n=562 on 4 GiB\""), std::string::npos);
    // The absent members are omitted, never null.
    EXPECT_EQ(json.find("explicit_builder"), std::string::npos);
    EXPECT_EQ(json.find("\"warning\""), std::string::npos);
}

TEST(ResultJsonTest, SelectionBlockCarriesExplicitBuilderAndWarning) {
    // An explicit builder that the heuristic overrides: the record
    // names both the input's choice and the divergence warning.
    qcx::io::RunResult result;
    qcx::io::RunSelection selection;
    selection.builder = qcx::io::BuilderKind::kDirect;
    selection.reasoning = "direct below the gpu crossover; explicit gpu honored";
    selection.explicitBuilder = qcx::io::BuilderKind::kGpu;
    selection.warning = "fock_builder = \"gpu\" fell back to direct";
    result.resourcesResolved.selection = std::move(selection);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"builder\": \"direct\""), std::string::npos);
    EXPECT_NE(json.find("\"explicit_builder\": \"gpu\""), std::string::npos);
    EXPECT_NE(json.find("\"warning\": \"fock_builder = \\\"gpu\\\" fell back to direct\""),
              std::string::npos);
    // No heuristic pick was made for an explicit builder.
    EXPECT_EQ(json.find("\"picked\""), std::string::npos);
}

TEST(ResultJsonTest, SelectionBlockAbsentWhenUnset) {
    // The honesty policy: no selection, no member.
    qcx::io::RunResult result;
    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_EQ(json.find("selection"), std::string::npos);
}

TEST(ResultJsonTest, SelectionBuilderMemberNamesTheMemberBesideTheFamilyWord) {
    // Schema 15, additive: the lean member's record keeps the FAMILY word
    // in `builder` ("direct") and names the member it
    // wired in `builder_member` ("lean"), so a consumer reads the choice
    // instead of inferring it from an absent explicit_builder.
    qcx::io::RunResult result;
    qcx::io::RunSelection selection;
    selection.builder = qcx::io::BuilderKind::kDirect;
    selection.picked = qcx::io::BuilderKind::kDirect;
    selection.builderMember = "lean";
    selection.reasoning = "no fock_builder given: the run defaults to the direct family";
    result.resourcesResolved.selection = std::move(selection);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"builder\": \"direct\""), std::string::npos);
    EXPECT_NE(json.find("\"builder_member\": \"lean\""), std::string::npos);

    // The family field did NOT become the member word: the member name is
    // its own key, and the family word is untouched by the addition.
    EXPECT_EQ(json.find("\"builder\": \"lean\""), std::string::npos);

    // The lean request names no family word, so explicit_builder stays
    // absent - and the member name is exactly what carries the choice.
    EXPECT_EQ(json.find("explicit_builder"), std::string::npos);
}

TEST(ResultJsonTest, SelectionBuilderMemberAbsentWhenUnset) {
    // The null honesty policy on the new member too: a record that never
    // carried a member name emits no key, never an empty string.
    qcx::io::RunResult result;
    qcx::io::RunSelection selection;
    selection.builder = qcx::io::BuilderKind::kGpu;
    selection.reasoning = "an explicit gpu request";
    result.resourcesResolved.selection = std::move(selection);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"selection\""), std::string::npos);
    EXPECT_EQ(json.find("builder_member"), std::string::npos);
}

TEST(ResultJsonTest, WorkspaceBudgetBlockEmittedWhenPresent) {
    // The ri_j budget-path audit: the cap-minus-base capacity
    // and the engine's cumulative Create-time commit.
    qcx::io::RunResult result;
    result.resourcesResolved.workspaceBudget =
        qcx::io::RunWorkspaceBudget{14'064'000'000ULL, 3'120'000'000ULL};

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"workspace_budget\""), std::string::npos);
    EXPECT_NE(json.find("\"capacity_bytes\": 14064000000"), std::string::npos);
    EXPECT_NE(json.find("\"committed_bytes\": 3120000000"), std::string::npos);
}

TEST(ResultJsonTest, WorkspaceBudgetBlockAbsentWhenUnset) {
    // The legacy null-budget path (cap 0 or a cap at/below the modeled
    // base) records nothing.
    qcx::io::RunResult result;
    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_EQ(json.find("workspace_budget"), std::string::npos);
}

TEST(ResultJsonTest, ModeRecordBlockEmittedWhenPresent) {
    // The engine's Create-time mode decision: the rung, the firing
    // estimate terms, and the exclusions that fired (the evidence
    // record on the big ri_j runs).
    qcx::io::RunResult result;
    qcx::io::RunModeRecord mode;
    mode.mode = "kLightPath";
    mode.predictedBytes = 3'214'000'000ULL;
    mode.reservedBytes = 3'120'000'000ULL;
    mode.budgetBytes = 14'064'000'000ULL;
    mode.remainingAtDecision = 10'944'000'000ULL;
    // The disk rung's modeled on-disk payload row (the driver's disk-rung knob): the
    // driver's disk route sets it; engine-record producers leave it zero.
    mode.diskBytes = 4'360'373'952ULL;
    mode.tensorBytes = 4'741'000'000ULL;
    mode.riMatrixBytes = 302'000'000ULL;
    mode.chunkPairs = 512;
    mode.patternExcluded = true;
    mode.tensorExcluded = true;
    // The A3b class-path admission-gate fields (the pair-class root fix,
    // 2026-08-31): the table's Create-time charge and the disengagement
    // flag.
    mode.classTableBytes = 6'326'191'200ULL;
    mode.classPathDisengaged = true;
    // The clamp-origin mirror (schema 12): the k a fired run carried and the
    // clamp-origin team read - a team-clamped k = 5 run.
    mode.concurrentSlots = 5;
    mode.defaultTeamSize = 5;
    result.resourcesResolved.modeRecord = std::move(mode);

    // The UHF exchange half (N1, 2026-08-31): a UHF run's Fock builds from
    // two direct builders, and the exchange builder's admission evidence
    // serializes as a second block under its own key.
    qcx::io::RunModeRecord exchange;
    exchange.mode = "kLightPath";
    exchange.classTableBytes = 6'326'191'200ULL;
    exchange.classPathDisengaged = true;
    result.resourcesResolved.exchangeModeRecord = std::move(exchange);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"mode_record\""), std::string::npos);
    EXPECT_NE(json.find("\"exchange_mode_record\""), std::string::npos);
    EXPECT_NE(json.find("\"mode\": \"kLightPath\""), std::string::npos);
    EXPECT_NE(json.find("\"predicted_bytes\": 3214000000"), std::string::npos);
    EXPECT_NE(json.find("\"remaining_at_decision\": 10944000000"), std::string::npos);
    EXPECT_NE(json.find("\"disk_bytes\": 4360373952"), std::string::npos);
    EXPECT_NE(json.find("\"chunk_pairs\": 512"), std::string::npos);
    EXPECT_NE(json.find("\"pattern_excluded\": true"), std::string::npos);
    EXPECT_NE(json.find("\"tensor_excluded\": true"), std::string::npos);
    EXPECT_NE(json.find("\"class_table_bytes\": 6326191200"), std::string::npos);
    EXPECT_NE(json.find("\"class_path_disengaged\": true"), std::string::npos);
    EXPECT_NE(json.find("\"concurrent_slots\": 5"), std::string::npos);
    EXPECT_NE(json.find("\"default_team_size\": 5"), std::string::npos);
    // The forced-disk statement (schema 18) is present on EVERY mode
    // record: false on an engine rung decision, true only on the driver's
    // forced disk route - an absent member could not tell the two apart.
    EXPECT_NE(json.find("\"forced_disk\": false"), std::string::npos);
}

TEST(ResultJsonTest, ModeRecordBlockAbsentWhenUnset) {
    // No budget, no decision - absent, never a fabricated fast-path row.
    qcx::io::RunResult result;
    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_EQ(json.find("mode_record"), std::string::npos);
}

TEST(ResultJsonTest, RiJkModeBlockEmittedWhenPresent) {
    // The composed full-RI builder's own Create-time rung decision (schema
    // 30): its OWN rung vocabulary and term decomposition, which is why it is
    // a block of its own rather than a second writer of mode_record - the two
    // families' rung words name different rungs and their numbers different
    // allocations. Every member is written, so a reader adds the estimate up.
    qcx::io::RunResult result;
    qcx::io::RunRiJkMode mode;
    mode.rung = "kBlocked";
    mode.predictedBytes = 19'900'000ULL;
    mode.reservedBytes = 19'850'000ULL;
    mode.budgetBytes = 269'000'000ULL;
    mode.remainingAtDecision = 268'000'000ULL;
    mode.maxBatchBytes = 4'194'304ULL;
    mode.structuralBytes = 3'100'000ULL;
    mode.rootBytes = 1'089'288ULL;
    mode.tensorBytes = 19'849'248ULL;
    mode.occTransformBytes = 1'147'520ULL;
    mode.fockBytes = 161'376ULL;
    mode.arenaBytes = 4'096ULL;
    mode.sliceFunctions = 60;
    mode.sliceCount = 3;
    result.resourcesResolved.riJkMode = std::move(mode);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"ri_jk_mode\""), std::string::npos);
    EXPECT_NE(json.find("\"rung\": \"kBlocked\""), std::string::npos);
    EXPECT_NE(json.find("\"predicted_bytes\": 19900000"), std::string::npos);
    EXPECT_NE(json.find("\"reserved_bytes\": 19850000"), std::string::npos);
    EXPECT_NE(json.find("\"remaining_at_decision\": 268000000"), std::string::npos);
    EXPECT_NE(json.find("\"tensor_bytes\": 19849248"), std::string::npos);
    EXPECT_NE(json.find("\"occ_transform_bytes\": 1147520"), std::string::npos);
    EXPECT_NE(json.find("\"slice_functions\": 60"), std::string::npos);
    EXPECT_NE(json.find("\"slice_count\": 3"), std::string::npos);
    // The two words are distinct rungs of ONE ladder, so the record cannot be
    // read as an alias of the RI-J vocabulary.
    EXPECT_EQ(json.find("\"rung\": \"kFastPath\""), std::string::npos);
}

TEST(ResultJsonTest, RiJkModeBlockAbsentWhenUnset) {
    // No budget, no decision - absent, never a fabricated rung row (the
    // sibling mode_record rule): the no-budget path consults no budget and
    // takes the fast rung, so there is no decision to report.
    qcx::io::RunResult result;
    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_EQ(json.find("ri_jk_mode"), std::string::npos);
}

TEST(ResultJsonTest, RiTensorModeBlockPairsRequestedWithRan) {
    // The demotion contract's disclosure surface (schema 19): a request
    // that could not be honoured must be readable as BOTH sides - what was
    // asked for and what ran - because a demotion the record cannot show is
    // the silent substitution the contract forbids.
    qcx::io::RunResult demoted;
    qcx::io::RunRiTensorMode requested;
    requested.requested = "forced_disk";
    requested.resolved = "in_memory";
    requested.outcome = "demoted";
    requested.reason = "fock_builder = \"direct\" has no ri_j_link disk route";
    requested.forced = true;
    demoted.resourcesResolved.riTensorMode = std::move(requested);

    const std::string demotedJson = qcx::io::SerializeRunResultJson(demoted);
    EXPECT_NE(demotedJson.find("\"ri_tensor_mode\""), std::string::npos);
    EXPECT_NE(demotedJson.find("\"requested\": \"forced_disk\""), std::string::npos);
    EXPECT_NE(demotedJson.find("\"resolved\": \"in_memory\""), std::string::npos);
    EXPECT_NE(demotedJson.find("\"outcome\": \"demoted\""), std::string::npos);
    // The member that names the KEY the request came from (schema 22): the
    // force is [diagnostics] force_disk_ri, so the word "forced_disk" in
    // `requested` is the REQUEST's name and this is its source. It stays
    // true on a demoted run - it names the key, never the outcome, which
    // `outcome` states - because a reader must be able to tell a forced
    // request from an unforced one without the input file.
    EXPECT_NE(demotedJson.find("\"forced\": true"), std::string::npos);
    EXPECT_NE(demotedJson.find("\"reason\": \"fock_builder = \\\"direct\\\" has no"),
              std::string::npos);

    // An honoured request carries no reason (the null honesty policy: an
    // absent reason is never a fabricated "fine"), and an omitted input key
    // leaves the block absent altogether - it is not a request.
    qcx::io::RunResult honoured;
    qcx::io::RunRiTensorMode asked;
    asked.requested = "auto";
    asked.resolved = "in_memory";
    asked.outcome = "honoured";
    honoured.resourcesResolved.riTensorMode = std::move(asked);

    const std::string honouredJson = qcx::io::SerializeRunResultJson(honoured);
    EXPECT_NE(honouredJson.find("\"outcome\": \"honoured\""), std::string::npos);
    EXPECT_NE(honouredJson.find("\"forced\": false"), std::string::npos);
    EXPECT_EQ(honouredJson.find("\"reason\""), std::string::npos);

    qcx::io::RunResult omitted;
    const std::string omittedJson = qcx::io::SerializeRunResultJson(omitted);
    EXPECT_EQ(omittedJson.find("ri_tensor_mode"), std::string::npos);
}

TEST(ResultJsonTest, RiChunkBytesBlockTellsHonouredFromDropped) {
    // The disclosure arm of the key split (schema 23), and the shape it asks
    // of itself: a reader of the DOCUMENT must be able to tell
    // "the key was honoured" from "the key was dropped, and why" without the
    // command line or the input file. The key is consumed at exactly one site
    // (the ri_j_link disk rung), so the drop is the common case and it used to
    // be stated nowhere at all.
    //
    // Three states, and the discrimination is asserted on the EMITTED text
    // rather than on the struct: that is what a consumer reads. The third
    // state is the one that keeps "dropped" honest - an omitted key is not a
    // dropped request, and a block that could not be told from an omission
    // would be the fabricated record this schema layer exists to refuse.
    qcx::io::RunResult dropped;
    qcx::io::RunRiChunkBytes unused;
    unused.requestedBytes = 134217728;
    unused.outcome = "dropped";
    unused.reason = "fock_builder = \"direct\" has no disk rung, so the chunk-size hint was "
                    "dropped";
    dropped.resourcesResolved.riChunkBytes = std::move(unused);

    const std::string droppedJson = qcx::io::SerializeRunResultJson(dropped);
    EXPECT_NE(droppedJson.find("\"ri_chunk_bytes\""), std::string::npos);
    EXPECT_NE(droppedJson.find("\"requested_bytes\": 134217728"), std::string::npos);
    EXPECT_NE(droppedJson.find("\"outcome\": \"dropped\""), std::string::npos);
    EXPECT_NE(droppedJson.find("\"reason\": \"fock_builder = \\\"direct\\\" has no"),
              std::string::npos);
    // A drop states the drop; it never also claims the hint was used.
    EXPECT_EQ(droppedJson.find("\"outcome\": \"honoured\""), std::string::npos);

    // Honoured: the disk rung took the hint, so there is no reason to give -
    // and an absent reason is never a fabricated "fine" (the null honesty
    // policy the selection and ri_tensor_mode blocks follow).
    qcx::io::RunResult honoured;
    qcx::io::RunRiChunkBytes used;
    used.requestedBytes = 134217728;
    used.outcome = "honoured";
    honoured.resourcesResolved.riChunkBytes = std::move(used);

    const std::string honouredJson = qcx::io::SerializeRunResultJson(honoured);
    EXPECT_NE(honouredJson.find("\"outcome\": \"honoured\""), std::string::npos);
    EXPECT_NE(honouredJson.find("\"requested_bytes\": 134217728"), std::string::npos);
    EXPECT_EQ(honouredJson.find("\"reason\""), std::string::npos);
    EXPECT_EQ(honouredJson.find("\"outcome\": \"dropped\""), std::string::npos);

    // The two documents a reader must tell apart carry the same requested
    // size and differ in exactly the way the disclosure is about: the outcome
    // word, and whether a reason stands beside it.
    EXPECT_NE(honouredJson.find("\"ri_chunk_bytes\""), std::string::npos);
    EXPECT_EQ(CountOccurrences(droppedJson, "\"outcome\": \"dropped\""), 1);
    EXPECT_EQ(CountOccurrences(honouredJson, "\"outcome\": \"dropped\""), 0);
    EXPECT_EQ(CountOccurrences(honouredJson, "\"outcome\": \"honoured\""), 1);

    // And the omitted key: no block at all. A run that never asked for a
    // chunk size has nothing to disclose, and the reader can tell that state
    // from the drop - which is the whole point of the block being optional.
    qcx::io::RunResult omitted;
    const std::string omittedJson = qcx::io::SerializeRunResultJson(omitted);
    EXPECT_EQ(omittedJson.find("ri_chunk_bytes"), std::string::npos);
}

TEST(ResultJsonTest, RiOrbitExpansionBlockSeparatesEngagedInertAndOmitted) {
    // The disclosure arm of the orbit-expansion key (schema 26). It has
    // one job the sibling blocks do not: the mechanism's engine-side permission
    // defaults ON, so a reader who sees no reduction in the input cannot infer
    // whether the expansion ran from the option's default - only from this
    // block, and only on the family that consumes the key.
    //
    // Four states, asserted on the EMITTED text (what a consumer reads): asked
    // and engaged, asked on a trivial group and inert, named-and-false, and
    // never named at all. The last two are different facts about the INPUT, and
    // a block that could not tell them apart would make a false request
    // indistinguishable from an omitted key.
    qcx::io::RunResult engaged;
    qcx::io::RunRiOrbitExpansion ran;
    ran.requested = true;
    ran.outcome = "engaged";
    engaged.resourcesResolved.riOrbitExpansion = std::move(ran);

    const std::string engagedJson = qcx::io::SerializeRunResultJson(engaged);
    EXPECT_NE(engagedJson.find("\"ri_orbit_expansion\""), std::string::npos);
    EXPECT_NE(engagedJson.find("\"requested\": true"), std::string::npos);
    EXPECT_NE(engagedJson.find("\"outcome\": \"engaged\""), std::string::npos);
    // Nothing to explain on the honoured arm: an absent reason is never a
    // fabricated "fine" (the null honesty policy).
    EXPECT_EQ(engagedJson.find("\"reason\""), std::string::npos);

    // Asked, both reductions built, and the group is trivial: the request
    // stands and the mechanism did nothing, which is a fact about the molecule
    // and must be stated rather than left as a silent non-event.
    qcx::io::RunResult inert;
    qcx::io::RunRiOrbitExpansion trivial;
    trivial.requested = true;
    trivial.outcome = "inert_trivial_group";
    trivial.reason = "the point group's order is 1 on the orbital or the auxiliary basis";
    inert.resourcesResolved.riOrbitExpansion = std::move(trivial);

    const std::string inertJson = qcx::io::SerializeRunResultJson(inert);
    EXPECT_NE(inertJson.find("\"outcome\": \"inert_trivial_group\""), std::string::npos);
    EXPECT_NE(inertJson.find("\"reason\": \"the point group"), std::string::npos);
    EXPECT_EQ(inertJson.find("\"outcome\": \"engaged\""), std::string::npos);

    // Named and false: the plain walk ran because the input asked for it, not
    // because anything refused - the distinction the `requested` member exists
    // for.
    qcx::io::RunResult plain;
    qcx::io::RunRiOrbitExpansion declined;
    declined.requested = false;
    declined.outcome = "not_requested";
    plain.resourcesResolved.riOrbitExpansion = std::move(declined);

    const std::string plainJson = qcx::io::SerializeRunResultJson(plain);
    EXPECT_NE(plainJson.find("\"requested\": false"), std::string::npos);
    EXPECT_NE(plainJson.find("\"outcome\": \"not_requested\""), std::string::npos);
    EXPECT_EQ(plainJson.find("\"reason\""), std::string::npos);

    // And the omitted key: no block at all, so the reader can tell "never
    // asked" from "asked and declined".
    qcx::io::RunResult omitted;
    const std::string omittedJson = qcx::io::SerializeRunResultJson(omitted);
    EXPECT_EQ(omittedJson.find("ri_orbit_expansion"), std::string::npos);
}

TEST(ResultJsonTest, EriStoreBlockDisclosesWhatRanNotWhatWasAsked) {
    // The disclosure arm of the disk-tier ERI store key (schema 33). Its job
    // is the one the engine-decorator seam could not do for itself: the store
    // is chosen OUTSIDE the builder, so the run's own JSON is the only
    // place a consumer can learn whether the request was honoured. A silent
    // substitution - the in-memory tier running under a disk request - is the
    // defect this block exists to make unreportable.
    //
    // Three states, asserted on the EMITTED text (what a consumer reads):
    // honoured, demoted with the store's own error, and never named at all.
    qcx::io::RunResult honoured;
    qcx::io::RunEriStore served;
    served.path = "run-eri-store.h5";
    served.engaged = "disk";
    served.demoted = false;
    served.hitQuartets = 4096;
    served.missQuartets = 512;
    served.readMs = 12.5;
    served.recomputeMs = 33.25;
    served.hitQuartetsByClass = {{1, 0, 3072}, {1, 1, 1024}};
    honoured.resourcesResolved.eriStore = std::move(served);

    const std::string honouredJson = qcx::io::SerializeRunResultJson(honoured);
    EXPECT_NE(honouredJson.find("\"eri_store\""), std::string::npos);
    EXPECT_NE(honouredJson.find("\"path\": \"run-eri-store.h5\""), std::string::npos);
    EXPECT_NE(honouredJson.find("\"engaged\": \"disk\""), std::string::npos);
    EXPECT_NE(honouredJson.find("\"demoted\": false"), std::string::npos);
    EXPECT_NE(honouredJson.find("\"hit_quartets\": 4096"), std::string::npos);
    EXPECT_NE(honouredJson.find("\"miss_quartets\": 512"), std::string::npos);
    EXPECT_NE(honouredJson.find("\"read_ms\": 12.5"), std::string::npos);
    EXPECT_NE(honouredJson.find("\"recompute_ms\": 33.25"), std::string::npos);
    // The per-class rows: the class each count belongs to is written beside
    // the count, so no key-encoding convention stands between a reader and the
    // number.
    EXPECT_EQ(CountOccurrences(honouredJson, "\"hit_quartets_by_class\": ["), 1);
    EXPECT_NE(honouredJson.find("\"l_bra\": 1"), std::string::npos);
    EXPECT_NE(honouredJson.find("\"l_ket\": 0"), std::string::npos);
    EXPECT_NE(honouredJson.find("\"hit_quartets\": 3072"), std::string::npos);
    // Nothing to explain on the honoured arm: an absent reason is never a
    // fabricated "fine" (the null honesty policy). Note this pins
    // `demoted_reason` and not `demoted`, which is always written.
    EXPECT_EQ(honouredJson.find("demoted_reason"), std::string::npos);

    // The demoted arm: the store could not be used, the run COMPLETED on the
    // closest workable arrangement, and the record says BOTH - the path that
    // was asked for and the reason the store's own error gave. A run whose
    // record cannot show the demotion is the substitution this block forbids.
    qcx::io::RunResult demoted;
    qcx::io::RunEriStore fellBack;
    fellBack.path = "run-eri-store.h5";
    fellBack.engaged = "in_memory_cache";
    fellBack.demoted = true;
    fellBack.demotedReason = "failed to open the ERI store: the path is a directory";
    demoted.resourcesResolved.eriStore = std::move(fellBack);

    const std::string demotedJson = qcx::io::SerializeRunResultJson(demoted);
    EXPECT_NE(demotedJson.find("\"path\": \"run-eri-store.h5\""), std::string::npos);
    EXPECT_NE(demotedJson.find("\"engaged\": \"in_memory_cache\""), std::string::npos);
    EXPECT_NE(demotedJson.find("\"demoted\": true"), std::string::npos);
    EXPECT_NE(demotedJson.find("\"demoted_reason\": \"failed to open the ERI store"),
              std::string::npos);
    // The counters are the decorator's and the store never served: zero, never
    // a fabricated reading, and never a substitute for the `engaged` word.
    EXPECT_NE(demotedJson.find("\"hit_quartets\": 0"), std::string::npos);
    // An empty per-class array is omitted rather than written as an empty one
    // (the block's own absence rule applied one level down).
    EXPECT_EQ(demotedJson.find("hit_quartets_by_class"), std::string::npos);

    // And the omitted key: no block at all, so "never asked" cannot be read as
    // "asked and demoted".
    qcx::io::RunResult omitted;
    const std::string omittedJson = qcx::io::SerializeRunResultJson(omitted);
    EXPECT_EQ(omittedJson.find("eri_store"), std::string::npos);
}

TEST(ResultJsonTest, SymmetryBlockEmittedWhenPresent) {
    // The full-group labeling record: the detected group, the
    // Abelian reduction, the per-MO labels, and the degenerate-subspace
    // records.
    qcx::io::RunResult result;
    qcx::io::RunSymmetry symmetry;
    symmetry.fullGroup = "C2v";
    symmetry.abelianReduction = "C2v";
    symmetry.labels = {"A1", "B2", "A1", "B1", "A2"};
    symmetry.irrepIndices = {0, 1, 2, 3, 4};
    symmetry.canonicalized = {qcx::io::RunDegenerateSubspace{"E", {1, 2}}};
    symmetry.straddled = {qcx::io::RunDegenerateSubspace{"E", {3, 4}}};
    symmetry.symmetrizationSubset = {"E"};
    symmetry.averagedElementCount = 2;
    result.symmetry = std::move(symmetry);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"symmetry\""), std::string::npos);
    EXPECT_NE(json.find("\"full_group\": \"C2v\""), std::string::npos);
    EXPECT_NE(json.find("\"abelian_reduction\": \"C2v\""), std::string::npos);
    EXPECT_NE(json.find("\"labels\""), std::string::npos);
    EXPECT_NE(json.find("\"irrep_indices\""), std::string::npos);
    EXPECT_NE(json.find("\"canonicalized\""), std::string::npos);
    EXPECT_NE(json.find("\"irrep_label\": \"E\""), std::string::npos);
    // The mo_indices arrays are pretty-printed multi-line (indent 2), so
    // the pin is key-count, not the formatted array: one record in each
    // of the two lists.
    EXPECT_EQ(CountOccurrences(json, "\"mo_indices\""), 2);
    EXPECT_NE(json.find("\"straddled\""), std::string::npos);
    EXPECT_NE(json.find("\"symmetrization_subset\""), std::string::npos);
    EXPECT_NE(json.find("\"averaged_element_count\": 2"), std::string::npos);
}

TEST(ResultJsonTest, SymmetryBetaBlockEmittedWhenPresent) {
    // UHF: the beta channel gets its own block.
    qcx::io::RunResult result;
    qcx::io::RunSymmetry symmetry;
    symmetry.fullGroup = "D2h";
    symmetry.abelianReduction = "D2h";
    result.symmetryBeta = std::move(symmetry);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"symmetry_beta\""), std::string::npos);
    EXPECT_NE(json.find("\"full_group\": \"D2h\""), std::string::npos);
}

TEST(ResultJsonTest, SymmetryBlocksAbsentWhenUnset) {
    // A C1 molecule, the [symmetry] full_group = false switch, or an
    // unrealizable group: no block - never a fabricated C1 record.
    qcx::io::RunResult result;
    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_EQ(json.find("symmetry"), std::string::npos);
}

// A symmetry-blocking disclosure built by hand rather than through the driver:
// the writer's contract is what this file pins, and it must hold independently
// of which arm of the scf guard produced the report. The numbers are the
// stretched-H2/STO-3G demotion the guard measured (3 blocked / 15 plain, one
// generator exactly adapted and three broken at 1.909).
qcx::io::RunSymmetryBlocking DemotedBlocking() {
    qcx::io::RunSymmetryBlocking blocking;
    blocking.action = "kDemoted";
    blocking.blockedSolveCount = 3;
    blocking.plainSolveCount = 15;
    blocking.generatorCommutatorNorms = {1.909, 1.909, 0.0, 1.909};
    blocking.maxGeneratorCommutatorNorm = 1.909;
    blocking.tolerance = 1e-08;
    return blocking;
}

TEST(ResultJsonTest, DemotedSymmetryBlockingIsDisclosed) {
    // The guard's decision (schema 32). A run that walked BOTH paths
    // must say so in its own record: the action word beside the two solve
    // counts, and the per-generator commutator norms of the worst measurement,
    // which are what let a reader VERIFY the refusal instead of taking it. A
    // demotion disclosed only to a caller holding the in-process result left
    // the run's JSON silent - the defect this block closes.
    qcx::io::RunResult result;
    result.converged = true;
    result.symmetryBlockingAlpha = DemotedBlocking();

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"symmetry_blocking\""), std::string::npos);
    EXPECT_NE(json.find("\"action\": \"kDemoted\""), std::string::npos);
    EXPECT_NE(json.find("\"blocked_solve_count\": 3"), std::string::npos);
    EXPECT_NE(json.find("\"plain_solve_count\": 15"), std::string::npos);
    // The number the decision was taken on travels with it, and the threshold
    // it was compared against travels beside it: without the tolerance the
    // norm is a number no reader can judge.
    EXPECT_NE(json.find("\"max_generator_commutator_norm\": 1.909"), std::string::npos);
    EXPECT_NE(json.find("\"tolerance\": 1e-08"), std::string::npos);
    // One per generator of the computational group, so the recorded maximum is
    // EXPLAINED by the recorded vector rather than asserted beside it: the
    // three broken generators carry the same norm as the maximum, and the
    // fourth - the generator this Fock stayed adapted to - carries 0. The list
    // is dumped multi-line (dump(2)), so the pin is the value count: 4 is the
    // three norms plus the maximum, and a dropped vector would leave 1.
    EXPECT_EQ(CountOccurrences(json, "\"generator_commutator_norms\": ["), 1);
    EXPECT_EQ(CountOccurrences(json, "1.909"), 4)
        << "the per-generator norms must reach the record, not just their maximum";
    // The beta channel disclosed nothing on this run: null INSIDE a present
    // block, never a second fabricated report - so one spin's demotion is
    // still readable when the other spin's Fock stayed adapted.
    EXPECT_NE(json.find("\"beta\": null"), std::string::npos);
}

TEST(ResultJsonTest, SymmetryBlockingDistinguishesUsedFromDemoted) {
    // The two states a reader must be able to tell apart are "the blocking was
    // used" and "the blocking was walked back mid-run". The counts are NOT what
    // separates them - a run can end with a positive blocked count and a
    // positive plain count either way - so this pins the action as the
    // disclosure: the record below carries the SAME counts and the SAME norms
    // as the demoted one, and differs from it in the action word alone. A
    // writer that re-derived the verdict from the counts (blocked > 0 and
    // plain > 0 => kDemoted) would emit the same JSON for both and fail here.
    qcx::io::RunResult used;
    used.converged = true;
    qcx::io::RunSymmetryBlocking usedBlocking;
    usedBlocking.action = "kUsed";
    usedBlocking.blockedSolveCount = 3;
    usedBlocking.plainSolveCount = 15;
    usedBlocking.generatorCommutatorNorms = {1.909, 1.909, 0.0, 1.909};
    usedBlocking.maxGeneratorCommutatorNorm = 1.909;
    usedBlocking.tolerance = 1e-08;
    used.symmetryBlockingAlpha = usedBlocking;
    used.symmetryBlockingBeta = usedBlocking;

    const std::string usedJson = qcx::io::SerializeRunResultJson(used);
    EXPECT_NE(usedJson.find("\"action\": \"kUsed\""), std::string::npos);
    EXPECT_EQ(usedJson.find("kDemoted"), std::string::npos);
    // Both channels disclosed: two reports, each carrying its own words.
    EXPECT_EQ(CountOccurrences(usedJson, "\"action\": \"kUsed\""), 2);

    qcx::io::RunResult demoted;
    demoted.converged = true;
    demoted.symmetryBlockingAlpha = DemotedBlocking();
    const std::string demotedJson = qcx::io::SerializeRunResultJson(demoted);
    EXPECT_NE(demotedJson.find("\"action\": \"kDemoted\""), std::string::npos);
    // The counts and norms are the same on both sides, so the words are the
    // only difference between the two records - which is the whole point of
    // carrying the action rather than inferring it.
    EXPECT_EQ(CountOccurrences(usedJson, "\"blocked_solve_count\": 3"), 2);
    EXPECT_EQ(CountOccurrences(demotedJson, "\"blocked_solve_count\": 3"), 1);
    EXPECT_EQ(CountOccurrences(demotedJson, "\"plain_solve_count\": 15"), 1);
    EXPECT_NE(usedJson, demotedJson);
}

TEST(ResultJsonTest, SymmetryBlockingIsAbsentOnARunThatNeverAsked) {
    // A run that never requested the blocking carries NOTHING: no basis set
    // supplied, a C1 molecule, an unrealizable group - the guard's own
    // kNotRequested, which the record states by absence rather than by a word
    // (so no consumer can read "blocking was used" off such a run). The
    // overlap-removal disclosure is filled here on purpose: an SCF loop DID
    // run and the result IS a UHF-shaped record, and the block is still
    // absent, because nothing about this run asked for the blocks.
    qcx::io::RunResult result;
    result.converged = true;
    result.numRemovedOverlapDirections = 0u;

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_EQ(json.find("symmetry_blocking"), std::string::npos);
    EXPECT_EQ(json.find("kNotRequested"), std::string::npos)
        << "the absence is the statement: a kNotRequested word would put a "
           "guard decision on a run that was never in a position to block";
}

TEST(ResultJsonTest, UnavailableSymmetryBlockingDoesNotReadAsUsed) {
    // kUnavailable is the linear-dependence removal's non-square
    // orthogonalizer: no blocked solve could be served, so NO Fock was
    // measured and no generator norm exists. It must not read as kUsed, and
    // its norms must not be fabricated - the empty list is the measurement
    // that was never taken (the null-vs-fabricated-zero policy).
    qcx::io::RunResult result;
    result.converged = true;
    qcx::io::RunSymmetryBlocking unavailable;
    unavailable.action = "kUnavailable";
    unavailable.blockedSolveCount = 0;
    unavailable.plainSolveCount = 4;
    result.symmetryBlockingAlpha = unavailable;

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"action\": \"kUnavailable\""), std::string::npos);
    EXPECT_EQ(json.find("kUsed"), std::string::npos);
    EXPECT_NE(json.find("\"generator_commutator_norms\": []"), std::string::npos);
    EXPECT_NE(json.find("\"max_generator_commutator_norm\": 0.0"), std::string::npos);
}

TEST(ResultJsonTest, CoefficientsBlockEmittedWhenPresent) {
    // The coefficients provenance block (schema 11): the loaded file's version, the
    // run's lookup context, every candidate's score in rank order with its
    // source, and the matched cell of the best selectable candidate.
    qcx::io::RunResult result;
    qcx::io::RunSelection selection;
    selection.builder = qcx::io::BuilderKind::kDirect;
    selection.picked = qcx::io::BuilderKind::kDirect;
    selection.reasoning = "table: direct 2.900 s vs ri_j_link 43.7 s";
    qcx::io::RunSelectionCoefficients coefficients;
    coefficients.fileVersion = "v1";
    coefficients.machineClassKey = "m1";
    coefficients.basisFamilyId = "ccpvdz";
    coefficients.presetId = "default";
    coefficients.cellKey = "direct_screened@m1@ccpvdz[100:599]@default@t[1:16]";
    coefficients.scores = {
        {"direct", 2.9, "table", true},
        {"ri_j_link", 43.7, "analytic", true},
        {"qfmm", 5.002, "analytic", true},
    };
    selection.coefficients = std::move(coefficients);
    result.resourcesResolved.selection = std::move(selection);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"coefficients\""), std::string::npos);
    EXPECT_NE(json.find("\"file_version\": \"v1\""), std::string::npos);
    EXPECT_NE(json.find("\"machine_class_key\": \"m1\""), std::string::npos);
    EXPECT_NE(json.find("\"basis_family_id\": \"ccpvdz\""), std::string::npos);
    EXPECT_NE(json.find("\"preset_id\": \"default\""), std::string::npos);
    EXPECT_NE(json.find("\"cell_key\""), std::string::npos);
    // A table pick is not borrowed: no inflation member.
    EXPECT_NE(json.find("\"borrowed\": false"), std::string::npos);
    EXPECT_EQ(json.find("\"inflation\""), std::string::npos);
    EXPECT_NE(json.find("\"candidate\": \"direct\""), std::string::npos);
    EXPECT_NE(json.find("\"score_seconds\": 2.9"), std::string::npos);
    EXPECT_EQ(CountOccurrences(json, "\"source\""), 3);
    EXPECT_NE(json.find("\"source\": \"table\""), std::string::npos);
    EXPECT_NE(json.find("\"selectable\": true"), std::string::npos);
}

TEST(ResultJsonTest, CoefficientsBlockCarriesBorrowedPick) {
    // A borrowed cell as the best selectable candidate: borrowed = true and
    // the stored inflation are both in the record.
    qcx::io::RunResult result;
    qcx::io::RunSelection selection;
    selection.builder = qcx::io::BuilderKind::kGpu;
    selection.picked = qcx::io::BuilderKind::kGpu;
    selection.reasoning = "borrowed: ri_j's cell inflated to the gpu family";
    qcx::io::RunSelectionCoefficients coefficients;
    coefficients.fileVersion = "v1";
    coefficients.machineClassKey = "m1";
    coefficients.basisFamilyId = "ccpvdz";
    coefficients.presetId = "default";
    coefficients.cellKey = "gpu@m1@ccpvdz[100:599]@default@t[1:16]";
    coefficients.borrowed = true;
    coefficients.inflation = 1.7;
    coefficients.scores = {
        {"gpu", 0.85, "borrowed", true},
        {"direct", 2.9, "table", true},
    };
    selection.coefficients = std::move(coefficients);
    result.resourcesResolved.selection = std::move(selection);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"borrowed\": true"), std::string::npos);
    EXPECT_NE(json.find("\"inflation\": 1.7"), std::string::npos);
    EXPECT_NE(json.find("\"source\": \"borrowed\""), std::string::npos);
}

TEST(ResultJsonTest, CoefficientsBlockAbsentWhenUnset) {
    // No coefficient file loaded for the run: the block is absent - absence
    // IS the record's "none" (the seam sits inert).
    qcx::io::RunResult result;
    qcx::io::RunSelection selection;
    selection.builder = qcx::io::BuilderKind::kDirect;
    selection.reasoning = "direct fastest on 4 GiB";
    result.resourcesResolved.selection = std::move(selection);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_EQ(json.find("coefficients"), std::string::npos);
}

TEST(ResultJsonTest, ComputeProfileBlockEmittedWhenPresent) {
    // The compute-profile block (schema 20): the hardware reading
    // the certified fp32 lane's default was RESOLVED against, with the
    // verdict and the threshold it was compared against, so the decision is
    // auditable without the reader re-deriving the rule. Every number is
    // emitted, not only the ones that fired - a passing gate still publishes
    // its readings, which is why the five members below that no verdict
    // depends on (`fp32_gflops`, `fp64_gflops`, `measured`, `simd_lane`,
    // `ratio_spread`) are pinned here and not left to the driver pin: the
    // unit-level record contract is this suite's own, and it is the only
    // place the block is checked without a Release-only heavy run.
    qcx::io::RunResult result;
    qcx::io::RunComputeProfile profile;
    profile.source = "host";
    profile.fp32Gflops = 118.6732;
    profile.fp64Gflops = 57.6105;
    profile.ratio = 2.0599;
    profile.measured = true;
    profile.simdLane = true;
    profile.pairs = 15;
    profile.ratioSpread = 0.4913;
    profile.certifiedLaneDefault = false;
    profile.certifiedLaneMinRatio = 4.0;
    result.resourcesResolved.computeProfile = std::move(profile);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"compute_profile\""), std::string::npos);
    EXPECT_NE(json.find("\"source\": \"host\""), std::string::npos);
    EXPECT_NE(json.find("\"fp32_gflops\": 118.6732"), std::string::npos);
    EXPECT_NE(json.find("\"fp64_gflops\": 57.6105"), std::string::npos);
    EXPECT_NE(json.find("\"ratio\": 2.0599"), std::string::npos);
    EXPECT_NE(json.find("\"measured\": true"), std::string::npos);
    EXPECT_NE(json.find("\"simd_lane\": true"), std::string::npos);
    // The delimiter is part of this pin: `"pairs": 15` is a PREFIX of
    // `"pairs": 15.0`, so a plain find would let the count drift from an
    // integer to a double - the type change the reader would silently
    // accept - and stay green.
    EXPECT_NE(json.find("\"pairs\": 15,"), std::string::npos);
    EXPECT_NE(json.find("\"ratio_spread\": 0.4913"), std::string::npos);
    EXPECT_NE(json.find("\"certified_lane_default\": false"), std::string::npos);
    // Schema 21: the forcing flag is ALWAYS present, on the
    // mode_record.forced_disk rule - an absent member could not distinguish a
    // forced run from a probed one - and it is false here, which is the
    // probed case this suite's block represents.
    EXPECT_NE(json.find("\"certified_lane_forced\": false"), std::string::npos);
    EXPECT_NE(json.find("\"certified_lane_min_ratio\": 4.0"), std::string::npos);
}

TEST(ResultJsonTest, ComputeProfileRecordsAForcedLaneBesideTheMachineVerdict) {
    // Schema 21: a run that FORCED the
    // certified fp32 lane records that fact BESIDE the machine's own verdict,
    // never instead of it. This is the pair a reader must not collapse: the
    // block still publishes what the probe measured and what its verdict
    // would have been (`certified_lane_default: false` here - the laptop
    // case, a host ratio of ~2 against the 4.0 threshold), and
    // `certified_lane_forced: true` says the lane's state did not come from
    // that verdict. A block without the flag on a forced run would read as a
    // measurement of a machine that never ran.
    qcx::io::RunResult result;
    qcx::io::RunComputeProfile profile;
    profile.source = "host";
    profile.fp32Gflops = 118.6732;
    profile.fp64Gflops = 57.6105;
    profile.ratio = 2.0599;
    profile.measured = true;
    profile.simdLane = true;
    profile.pairs = 15;
    profile.ratioSpread = 0.4913;
    profile.certifiedLaneDefault = false;
    profile.certifiedLaneForced = true;
    profile.certifiedLaneMinRatio = 4.0;
    result.resourcesResolved.computeProfile = std::move(profile);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"certified_lane_forced\": true"), std::string::npos);
    EXPECT_NE(json.find("\"certified_lane_default\": false"), std::string::npos);
    EXPECT_NE(json.find("\"ratio\": 2.0599"), std::string::npos);
}

TEST(ResultJsonTest, ComputeProfileUnmeasuredIsFalseNotAbsence) {
    // `measured` is what keeps a MEASURED 1.0 apart from the UNMEASURED 1.0:
    // a scalar machine really does measure ~1.0, and a device probe that
    // failed to measure reports the same 1.0 through the same block (the
    // documented unknown). Both blocks are present, so the block carries the
    // distinction rather than the reader having to infer it from a key that
    // is not there - and the device arm measures once, so its not-reported
    // pair count and spread are true zeros, not the host arm's borrowed
    // ones.
    qcx::io::RunResult result;
    qcx::io::RunComputeProfile profile;
    profile.source = "device";
    profile.measured = false;
    profile.certifiedLaneMinRatio = 4.0;
    result.resourcesResolved.computeProfile = std::move(profile);

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"compute_profile\""), std::string::npos);
    EXPECT_NE(json.find("\"source\": \"device\""), std::string::npos);
    EXPECT_NE(json.find("\"ratio\": 1.0"), std::string::npos);
    EXPECT_NE(json.find("\"measured\": false"), std::string::npos);
    EXPECT_NE(json.find("\"simd_lane\": false"), std::string::npos);
    EXPECT_NE(json.find("\"fp32_gflops\": 0.0"), std::string::npos);
    EXPECT_NE(json.find("\"fp64_gflops\": 0.0"), std::string::npos);
    EXPECT_NE(json.find("\"pairs\": 0,"), std::string::npos);
    EXPECT_NE(json.find("\"ratio_spread\": 0.0"), std::string::npos);
    EXPECT_NE(json.find("\"certified_lane_default\": false"), std::string::npos);
}

TEST(ResultJsonTest, ComputeProfileBlockAbsentWhenUnset) {
    // The honesty policy: a run that resolved no probe records no block -
    // absent, never a fabricated row carrying the struct's unfilled defaults
    // (the empty source, the unmeasured ratio 1.0), which would read as a
    // measurement of a scalar machine.
    qcx::io::RunResult result;
    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_EQ(json.find("compute_profile"), std::string::npos);
}

TEST(ResultJsonTest, ComputeProfileCreateRefusesABlockWithNoResolvedArm) {
    // The invariant the block was documented to satisfy and nothing enforced
    // (result_json.hpp RunComputeProfile::Create): `source` names the arm the
    // run RESOLVED, so a present block may only say "host" or "device". The
    // invariant survived only because this block had one producer - the check
    // is on the TYPE so a second one (a call site, a refactor, a fixture)
    // inherits it rather than reading about it.
    //
    // Both refusals here are states the schema doc never listed as values:
    // the empty default a filled-but-armless block carries, and "none" - the
    // member's own default before this check existed, which no producer ever
    // set. A present `"source": "none"` row would read as an arm that
    // produced the numbers; the no-probe case is the block's ABSENCE.
    qcx::io::RunComputeProfile unfilled;
    unfilled.certifiedLaneMinRatio = 4.0;
    const auto refused = qcx::io::RunComputeProfile::Create(std::move(unfilled));
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(refused.error().message.find("source"), std::string::npos);

    qcx::io::RunComputeProfile deadDefault;
    deadDefault.source = "none";
    deadDefault.certifiedLaneMinRatio = 4.0;
    const auto alsoRefused = qcx::io::RunComputeProfile::Create(std::move(deadDefault));
    ASSERT_FALSE(alsoRefused.has_value());
    EXPECT_EQ(alsoRefused.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(ResultJsonTest, ComputeProfileCreateRefusesAnUnsetThreshold) {
    // `certified_lane_min_ratio` is the one number in the block whose wrong
    // value flips a VERDICT rather than misdescribing a reading: 0.0 is the
    // "unset" sentinel and every ratio passes it, so an emitted 0.0 would
    // read as the certified fp32 lane enabled on hardware with no fp32
    // advantage at all - this box measures ~2 against the lane's 4.0, and the
    // lane is OFF there. A negative threshold is not a threshold either.
    qcx::io::RunComputeProfile unset;
    unset.source = "host";
    unset.ratio = 2.0599;
    // certifiedLaneMinRatio stays at the member's 0.0 default.
    const auto refused = qcx::io::RunComputeProfile::Create(std::move(unset));
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(refused.error().message.find("certified_lane_min_ratio"), std::string::npos);

    qcx::io::RunComputeProfile negative;
    negative.source = "host";
    negative.ratio = 2.0599;
    negative.certifiedLaneMinRatio = -4.0;
    EXPECT_FALSE(qcx::io::RunComputeProfile::Create(std::move(negative)).has_value());
}

TEST(ResultJsonTest, ComputeProfileCreateRefusesHostReadingsOnTheDeviceArm) {
    // The device arm's three not-reported readings, stated by both this
    // header's field docs and the schema doc and enforced by nothing
    // until now: the device probe measures once, so `simd_lane`, `pairs` and
    // `ratio_spread` stay at their not-reported values. Without the check a
    // device block could carry a host SIMD lane width and a host noise
    // estimate for a GPU, which is the same class of silent wrong answer as a
    // host reading arriving as a device one.
    for (const int which : {0, 1, 2})
    {
        qcx::io::RunComputeProfile device;
        device.source = "device";
        device.certifiedLaneMinRatio = 4.0;

        if (which == 0)
        {
            device.simdLane = true;
        } else if (which == 1)
        {
            device.pairs = 5;
        } else
        {
            device.ratioSpread = 0.4913;
        }

        const auto refused = qcx::io::RunComputeProfile::Create(std::move(device));
        ASSERT_FALSE(refused.has_value()) << "device arm reading " << which;
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
    }
}

TEST(ResultJsonTest, ComputeProfileCreateAcceptsBothArmsAndRewritesNothing) {
    // The guard is a filter, not a rewrite: both arms' honest shapes pass, and
    // what comes back IS the candidate - so a producer cannot end up emitting
    // a differently-filled block than the one that was validated.
    qcx::io::RunComputeProfile host;
    host.source = "host";
    host.fp32Gflops = 118.6732;
    host.fp64Gflops = 57.6105;
    host.ratio = 2.0599;
    host.measured = true;
    host.simdLane = true;
    host.pairs = 5;
    host.ratioSpread = 0.4913;
    host.certifiedLaneMinRatio = 4.0;
    const auto hostOk = qcx::io::RunComputeProfile::Create(std::move(host));
    ASSERT_TRUE(hostOk.has_value()) << hostOk.error().message;
    EXPECT_EQ(hostOk->source, "host");
    EXPECT_DOUBLE_EQ(hostOk->ratio, 2.0599);
    EXPECT_DOUBLE_EQ(hostOk->certifiedLaneMinRatio, 4.0);
    EXPECT_EQ(hostOk->pairs, 5);
    EXPECT_TRUE(hostOk->measured);

    // The unmeasured device row the serialization test above emits: the
    // device arm's failure case - no measurement, but an arm that APPLIED -
    // which is a block, not an absence.
    qcx::io::RunComputeProfile device;
    device.source = "device";
    device.certifiedLaneMinRatio = 4.0;
    const auto deviceOk = qcx::io::RunComputeProfile::Create(std::move(device));
    ASSERT_TRUE(deviceOk.has_value()) << deviceOk.error().message;
    EXPECT_EQ(deviceOk->source, "device");
    EXPECT_DOUBLE_EQ(deviceOk->ratio, 1.0);
    EXPECT_FALSE(deviceOk->measured);
}

} // namespace
