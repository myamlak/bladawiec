// The auxiliary-basis SELECTION notice disclosure (schema 25):
// RunApproximation::auxNotice and the `aux_notice` key it serializes as.
//
// This is the record-visible half of the tiered aux rule. The rule always
// produces a default, so a run whose aux selection lands in a region known to
// fit less reliably is demoted-to-a-default WITH disclosure rather than refused
// - and a demotion the record cannot show is not a demotion, it is the silent
// substitution this record exists to forbid. A notice the record cannot carry is
// "calculated and invisible", which is the same defect in another costume.
//
// Deliberately its own file rather than a cell in result_json_test.cpp: the
// emit path has its own presence rule (present when a notice exists, ABSENT
// when it does not - never null, which a consumer could not tell from
// "computed and empty"), and a reader looking for the notice's record contract
// should find it in one place.

#include "qcx/io/result_json.hpp"

#include <gtest/gtest.h>
#include <string>
#include <utility>

namespace {

// A run carrying an approximation block, with or without the notice. Built by
// hand rather than through the driver: the writer's contract is what this file
// pins, and it must hold independently of which arm of the driver filled it.
qcx::io::RunResult ResultWithApproximation(bool withNotice) {
    qcx::io::RunResult result;
    qcx::io::RunApproximation approximation;
    approximation.exchange = "occ_ri_k";
    approximation.auxBasis = "def2-universal-jkfit";

    if (withNotice)
    {
        approximation.auxNotice =
            "orbital basis \"sto-3g\" is in a region the RI fit serves less reliably";
    }

    qcx::io::RunSelection selection;
    selection.builder = qcx::io::BuilderKind::kRiJk;
    selection.reasoning = "explicit fock_builder \"ri_jk\" honored as requested";
    selection.approximation = std::move(approximation);
    result.resourcesResolved.selection = std::move(selection);
    return result;
}

} // namespace

TEST(ApproximationNoticeTest, NoticeIsSerializedWhenPresent) {
    const std::string json = qcx::io::SerializeRunResultJson(ResultWithApproximation(true));
    EXPECT_NE(json.find("\"aux_notice\""), std::string::npos) << json;
    EXPECT_NE(json.find("serves less reliably"), std::string::npos) << json;
}

TEST(ApproximationNoticeTest, NoticeKeyIsAbsentRatherThanNullWhenUnset) {
    // The null honesty policy: an unset notice omits the key. A null would be
    // indistinguishable from a notice that was computed and came out empty,
    // which is the distinction the whole block exists to keep.
    const std::string json = qcx::io::SerializeRunResultJson(ResultWithApproximation(false));
    EXPECT_EQ(json.find("aux_notice"), std::string::npos)
        << "an absent notice must omit the key rather than emit null: " << json;
    // And the block itself is still there - the notice is a field ON it, not a
    // replacement for it.
    EXPECT_NE(json.find("\"approximation\""), std::string::npos) << json;
}

TEST(ApproximationNoticeTest, BlockCarriesANoticeOverAnExactKernel) {
    // The rule: a notice must never hang off an ABSENT block, so a run
    // whose kernel is exact still emits the block when the aux selection warns.
    // `exchange` then reads "exact" - the meaning the absent case carries,
    // STATED rather than left for the reader to reconstruct.
    qcx::io::RunResult result = ResultWithApproximation(true);
    result.resourcesResolved.selection->approximation->exchange = "exact";

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"exchange\": \"exact\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"aux_notice\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"aux_basis\""), std::string::npos) << json;
}
