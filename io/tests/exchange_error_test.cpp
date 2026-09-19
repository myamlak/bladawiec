// The approximated-exchange MEASURED-error disclosure (schema 31):
// RunExchangeError and the `exchange_error` block it serializes as.
//
// This is the record half of the rule that governs the ri_jk path - the path
// closes with its error DISCLOSED - and it exists because a
// disclosure a run cannot emit is not a disclosure: the achieved per-atom error
// lived in a test cell's output and in prose, while the mapping beside it
// carried the BAR and no achieved value at all, so a reader of an energy built
// through an auxiliary fit had the fit's NAME and nothing to size that energy's
// error with, and not even the bar it is read against.
//
// Its own file rather than a cell in result_json_test.cpp, for the reason the
// notice's file states: the emit path has its own presence rule - present when
// the block's `exchange` names an APPROXIMATED contraction, ABSENT when it does
// not, never null - and a reader looking for this field's record contract
// should find it in one place.

#include "qcx/io/result_json.hpp"

#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace {

// The number following "key" in the dumped document (nlohmann's dump(2) writes
// `"key": value`), or nullopt when the key is absent. Asserted on the TEXT
// rather than on a parsed document because nlohmann is a PRIVATE dependency of
// qcx-io (io/CMakeLists.txt: implementation-only, no public header exposes it),
// so a test target carries no include path for it - the same reason
// result_json_test.cpp counts keys in the text.
std::optional<double> JsonNumber(const std::string& json, std::string_view key) {
    const auto keyPos = json.find(key);

    if (keyPos == std::string::npos)
    {
        return std::nullopt;
    }

    const auto valuePos = json.find(':', keyPos);

    if (valuePos == std::string::npos)
    {
        return std::nullopt;
    }

    return std::stod(json.substr(valuePos + 1));
}

// A run carrying the approximated-exchange block, with or without the measured
// error. Built by hand rather than through the driver: the writer's contract is
// what this file pins, and it must hold independently of which arm of the
// driver filled it.
qcx::io::RunResult ResultWithExchangeError(bool withError) {
    qcx::io::RunResult result;
    qcx::io::RunApproximation approximation;
    approximation.exchange = "occ_ri_k";
    approximation.auxBasis = "def2-universal-jkfit";

    if (withError)
    {
        // The values the driver fills: the path's worst characterized cell
        // (h2o_sto3g, def2-universal-jkfit) and the kLoose bar. Both are PER
        // ATOM - the cell's measured total 3.52432e-4 Eh is over three atoms,
        // and the bar is per atom, so the record must pair like with like.
        approximation.exchangeError = qcx::io::RunExchangeError{
            1.174773e-4, 7.97e-4, "kLoose", "h2o_sto3g", "def2-universal-jkfit"};
    }

    qcx::io::RunSelection selection;
    selection.builder = qcx::io::BuilderKind::kRiJk;
    selection.reasoning = "explicit fock_builder \"ri_jk\" honored as requested";
    selection.approximation = std::move(approximation);
    result.resourcesResolved.selection = std::move(selection);
    return result;
}

} // namespace

TEST(ExchangeErrorTest, CarriesBothNumbersAndTheScopeOfTheMeasuredOne) {
    // Both numbers are the disclosure: a bare measured value with no bar is a
    // number, not a judgement, and the bar without its preset is unattributable
    // (the record echoes [method] accuracy nowhere else).
    const std::string json = qcx::io::SerializeRunResultJson(ResultWithExchangeError(true));
    EXPECT_NE(json.find("\"exchange_error\""), std::string::npos) << json;

    const auto measured = JsonNumber(json, "\"measured_per_atom_hartree\"");
    const auto bar = JsonNumber(json, "\"bar_per_atom_hartree\"");
    ASSERT_TRUE(measured.has_value()) << json;
    ASSERT_TRUE(bar.has_value()) << json;
    EXPECT_DOUBLE_EQ(*measured, 1.174773e-4);
    EXPECT_DOUBLE_EQ(*bar, 7.97e-4);
    EXPECT_NE(json.find("\"bar_preset\": \"kLoose\""), std::string::npos) << json;

    // The scope of the measured value travels WITH it: the fixture and the fit
    // the cell measured on, so the path's number can never read as this run's
    // own error.
    EXPECT_NE(json.find("\"measured_on\": \"h2o_sto3g\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"measured_aux_basis\": \"def2-universal-jkfit\""), std::string::npos)
        << json;
}

TEST(ExchangeErrorTest, AbsentRatherThanNullWhenUnset) {
    // The null honesty policy the notice beside it uses: an unset member omits
    // the key, because a null would be indistinguishable from a computed value
    // that came out empty. The block itself is still emitted - this field is a
    // member ON it, not a replacement for it.
    const std::string json = qcx::io::SerializeRunResultJson(ResultWithExchangeError(false));
    EXPECT_EQ(json.find("\"exchange_error\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"exchange\": \"occ_ri_k\""), std::string::npos) << json;
}

TEST(ExchangeErrorTest, AbsentOnAnExactKernelBlockCarryingANotice) {
    // The narrow arm, and the one the field's presence rule exists for: the
    // block is emitted on an EXACT-kernel run when the aux selection warns
    // (schema 25), and on that run the fitted exchange's error is no error of
    // that run's - printing it would be a lie about the run rather than a
    // disclosure of it. The block stays; the measured error does not appear on
    // it.
    qcx::io::RunResult result = ResultWithExchangeError(false);
    result.resourcesResolved.selection->approximation->exchange = "exact";
    result.resourcesResolved.selection->approximation->auxNotice =
        "orbital basis \"sto-3g\" is in a region the RI fit serves less reliably";

    const std::string json = qcx::io::SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"exchange\": \"exact\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"aux_notice\""), std::string::npos) << json;
    EXPECT_EQ(json.find("\"exchange_error\""), std::string::npos) << json;
}
