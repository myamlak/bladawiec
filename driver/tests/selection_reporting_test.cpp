// The selection record the driver reports in resources_resolved under the
// owner's ruling of 2026-09-13 (the size ladder) - the
// wired builder, the pick the record reports, the override flag and the
// warning - through the full driver path (TOML -> RunInput -> resolution
// -> JSON). The no-builder default is the size ladder: the direct family
// lean member at nBasis <= 1000, ri_j_link to 2000, qfmm above. The
// H2-class fixtures here are all in the bottom tier, so they wire the lean
// member; the family word stays "direct" and the record names the member it
// actually wired in builder_member (- the field the consistency
// test at the end of this file pins against the family word). The
// assertions never depend on whether
// a CUDA device is present, except where a test is gated on the build's
// QcxHasCuda macro on purpose (the device-less fallback pair).
//
// The fixtures' [scf] block names the OPERATING DEFAULT (energy 1e-8,
// density 1e-6). It used to pin 1e-10/1e-10 - a convention inherited
// when the block was copied, never a property these
// tests assert: their subject is the BUILDER SELECTION the driver reports in
// resources_resolved, and no assertion below reads an energy or an iteration
// count. The owner's 2026-09-15 ruling retired 1e-10/1e-12 outright, so the
// inherited gate went with it (tools/check_gate_tolerances.py, which fails an
// undeclared tight gate, is the guard that keeps it gone). [method].accuracy
// is untouched: that preset selects SCREENING budgets, not the gate.

#include "fast_test_mode.hpp"
#include "qcx/driver/run_driver.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/io/result_json.hpp"

#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace {

using qcx::driver::RunDriver;
using qcx::io::ParseRunInput;

// The H2 geometry of the driver pin without [method].fock_builder: the
// key is absent, so the resolution runs the no-builder default - the
// direct family (its lean member at nBasis <= 1000, wired inside
// WireRhfFockBuilder).
constexpr const char* kH2AutoToml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["H", 0.0, 0.0, 0.0],
    ["H", 0.74084809526419992, 0.0, 0.0],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
accuracy = "kTight"

[scf]
max_iterations = 100
energy_tolerance = 1e-8
density_tolerance = 1e-6
use_diis = true
)";

// kH2AutoToml with the builder pinned to direct: the explicit path.
constexpr const char* kH2DirectToml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["H", 0.0, 0.0, 0.0],
    ["H", 0.74084809526419992, 0.0, 0.0],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kTight"

[scf]
max_iterations = 100
energy_tolerance = 1e-8
density_tolerance = 1e-6
use_diis = true
)";

// kH2AutoToml with the explicit within-family "lean" spelling: the same
// lean member the absent key reaches at this size, requested by name (the
// spelling is a within-family request, not a family word).
constexpr const char* kH2LeanToml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["H", 0.0, 0.0, 0.0],
    ["H", 0.74084809526419992, 0.0, 0.0],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "lean"
accuracy = "kTight"

[scf]
max_iterations = 100
energy_tolerance = 1e-8
density_tolerance = 1e-6
use_diis = true
)";

// kH2AutoToml with the builder pinned to gpu: the device-less fallback probe.
constexpr const char* kH2GpuToml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["H", 0.0, 0.0, 0.0],
    ["H", 0.74084809526419992, 0.0, 0.0],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "gpu"
accuracy = "kTight"

[scf]
max_iterations = 100
energy_tolerance = 1e-8
density_tolerance = 1e-6
use_diis = true
)";

// The O2 triplet of the UHF pin, auto builder: the v1 UHF path wires the
// direct builder only (SAD seed, per-spin DIIS - the established recipe).
constexpr const char* kO2UhfAutoToml = R"(
[molecule]
charge = 0
multiplicity = 3
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 1.2075000023889084],
]

[basis]
orbital = "sto-3g"

[method]
type = "uhf"
accuracy = "kTight"

[scf]
use_diis = true

[guess]
type = "sad"
)";

// Named RunInputText: "Run" alone collides with testing::Test::Run inside
// the TEST() bodies.
qcx::Result<std::string> RunInputText(const char* toml) {
    auto input = ParseRunInput(toml);

    if (!input.has_value())
    {
        return std::unexpected(input.error());
    }

    return RunDriver(*input);
}

// The position of the resources_resolved selection sub-block, or npos.
// The driver serializes the selection block as the LAST member of
// resources_resolved, so a probe from here can only match inside it.
std::size_t SelectionPos(const std::string& json) {
    return json.find("\"selection\":");
}

// The string value after the "key" member at or after \p from. The
// driver's selection values carry escaped quotes (the device-less fallback warning names
// fock_builder "gpu"), so the scan skips \" escapes. Returns nullopt when
// the key is absent or the value is null.
std::optional<std::string> JsonString(const std::string& json,
                                      std::string_view key,
                                      std::size_t from = 0) {
    const auto keyPos = json.find(key, from);

    if (keyPos == std::string::npos)
    {
        return std::nullopt;
    }

    const auto valuePos = json.find(':', keyPos);

    if (valuePos == std::string::npos)
    {
        return std::nullopt;
    }

    std::size_t start = valuePos + 1;

    while (start < json.size() && json[start] == ' ')
    {
        ++start;
    }

    if (start >= json.size() || json[start] != '"')
    {
        return std::nullopt;
    }

    ++start;
    std::string value;

    for (; start < json.size(); ++start)
    {
        if (json[start] == '\\' && start + 1 < json.size() && json[start + 1] == '"')
        {
            value.push_back('"');
            ++start;
        } else if (json[start] == '"')
        {
            return value;
        } else
        {
            value.push_back(json[start]);
        }
    }

    return std::nullopt;
}

} // namespace

TEST(SelectionReportingTest, AutoRunReportsTheSelectionBlock) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kH2AutoToml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto selection = SelectionPos(*result);
    ASSERT_NE(selection, std::string::npos);

    // The 2026-09-13 ladder: the no-builder run at H2 (2 basis functions,
    // the lean tier) wires the direct family lean member - the record
    // keeps the family name "direct" and names the member - the
    // record names the tier it resolved and the driver followed it, so
    // nothing diverged and no warning fires.
    EXPECT_EQ(JsonString(*result, "\"builder\":", selection), "direct");
    EXPECT_EQ(JsonString(*result, "\"builder_member\":", selection), "lean");
    EXPECT_EQ(JsonString(*result, "\"picked\":", selection), "direct");
    EXPECT_EQ(JsonString(*result, "\"explicit_builder\":", selection), std::nullopt);
    const auto reasoning = JsonString(*result, "\"reasoning\":", selection);
    ASSERT_TRUE(reasoning.has_value());
    EXPECT_NE(reasoning->find("defaults to the size ladder"), std::string::npos);
    // The chosen boundary says so where the policy is read (the owner
    // requirement): a reader of this record must not take 2000 for a
    // measured crossover.
    EXPECT_NE(reasoning->find("CHOSEN value with NO measurement"), std::string::npos);

    // The selection is the last member of resources_resolved: any
    // "warning" key after it would be the record's. An absent warning
    // serializes as no member at all.
    EXPECT_EQ(result->find("\"warning\":", selection), std::string::npos);

    // The schema version pin for the selection record's document: the
    // record entered at kSchemaVersion 7 (the auto-selection bump), and
    // the pin rides the io constant so later bumps - the calibration ri-j
    // counters (9), the disk-rung disk rung's mode_record rows (10) - cannot
    // drift this file again (7bc753b updated io/tests but missed it).
    EXPECT_NE(
        result->find("\"schema_version\": " + std::to_string(qcx::io::RunResult::kSchemaVersion)),
        std::string::npos);
}

TEST(SelectionReportingTest, ExplicitDirectRequestIsRecordedSilently) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kH2DirectToml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto selection = SelectionPos(*result);
    ASSERT_NE(selection, std::string::npos);

    // The explicit request is honored verbatim (Ruling B: an explicit
    // "direct" keeps the machinery below, not the lean default): recorded
    // as explicit, the member name is the family word itself (the
    // machinery, never the lean member), the record's pick stays the
    // no-builder default (the direct family), and no warning fires - the
    // explicit-override divergence warnings died with the heuristic pick.
    EXPECT_EQ(JsonString(*result, "\"builder\":", selection), "direct");
    EXPECT_EQ(JsonString(*result, "\"builder_member\":", selection), "direct");
    EXPECT_EQ(JsonString(*result, "\"picked\":", selection), "direct");
    EXPECT_EQ(JsonString(*result, "\"explicit_builder\":", selection), "direct");
    EXPECT_EQ(result->find("\"warning\":", selection), std::string::npos);
}

#if !defined(QcxHasCuda)
TEST(SelectionReportingTest, DeviceLessGpuRequestFallsBackAndReportsIt) {
    // The retargeted the device-less fallback behavior through the driver path: an explicit
    // "gpu" without a device falls back to the LADDER's own tier for this
    // run - the direct family lean member at H2, 2 basis functions (the
    // tier the run would have wired with no key at all; never a model pick,
    // the model auto-selection is retired) - and the record carries the
    // loud warning. The CUDA build defines QcxHasCuda, so this case only
    // runs on the default lanes; the fallback itself is what the no-device
    // lanes wire.
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kH2GpuToml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto selection = SelectionPos(*result);
    ASSERT_NE(selection, std::string::npos);

    EXPECT_EQ(JsonString(*result, "\"builder\":", selection), "direct");
    // The fallback target is the ladder's tier for this run, member
    // included: the request was for gpu, but what ran is the direct
    // family lean member, and the member name says so (the warning names
    // the divergence separately).
    EXPECT_EQ(JsonString(*result, "\"builder_member\":", selection), "lean");
    EXPECT_EQ(JsonString(*result, "\"picked\":", selection), "direct");
    EXPECT_EQ(JsonString(*result, "\"explicit_builder\":", selection), "gpu");
    const auto warning = JsonString(*result, "\"warning\":", selection);
    ASSERT_TRUE(warning.has_value());
    EXPECT_NE(warning->find("no CUDA device is present"), std::string::npos);
    EXPECT_NE(warning->find("falling back to the no-fock_builder size ladder's tier"),
              std::string::npos);
}
#endif

#if defined(QcxHasCuda)
TEST(SelectionReportingTest, CudaGpuRequestWiresGpuAndStaysSilent) {
    // The CUDA side of the device-less fallback pair: with a device present, the explicit
    // gpu request wires gpu verbatim (an explicit opt-in under the flip -
    // no fallback fires). The record carries the explicit flag and no
    // warning: the explicit-override divergence warning that used to name
    // the heuristic pick is retired with the model.
    auto result = RunInputText(kH2GpuToml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto selection = SelectionPos(*result);
    ASSERT_NE(selection, std::string::npos);

    EXPECT_EQ(JsonString(*result, "\"builder\":", selection), "gpu");
    EXPECT_EQ(JsonString(*result, "\"picked\":", selection), "direct");
    EXPECT_EQ(JsonString(*result, "\"explicit_builder\":", selection), "gpu");
    EXPECT_EQ(result->find("\"warning\":", selection), std::string::npos);
}
#endif

TEST(SelectionReportingTest, UhfAutoRunWiresTheLeanMember) {
    // UHF rides the same ladder as RHF, lean member included (the
    // Unrestricted UHF seam, 2026-09-11): the no-builder O2 sto-3g run at
    // nBasis <= 1000 wires the direct family's lean member, and the
    // record keeps the family name "direct" while naming the member
    //  - the same shape the RHF auto run reports. The record
    // stays silent: no divergence concept exists anymore at any topology.
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kO2UhfAutoToml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto selection = SelectionPos(*result);
    ASSERT_NE(selection, std::string::npos);

    EXPECT_EQ(JsonString(*result, "\"builder\":", selection), "direct");
    EXPECT_EQ(JsonString(*result, "\"builder_member\":", selection), "lean");
    EXPECT_EQ(JsonString(*result, "\"picked\":", selection), "direct");
    EXPECT_EQ(JsonString(*result, "\"explicit_builder\":", selection), std::nullopt);
    EXPECT_EQ(result->find("\"warning\":", selection), std::string::npos);
}

// The drift pin on those two records. `builder_member` and `builder` are
// two records
// of ONE choice - the member the run wired - and a consumer may read
// either, so they must agree on every run, whichever spelling requested
// the member. The invariant: builder_member == "lean" EXACTLY on the lean
// request (the lean member IS the direct family's within-family member);
// off that carve-out the member name equals the family word. Pinning the
// relation rather than the two literals separately is the point: a later
// member name that contradicts the family slot fails here even if every
// literal still looks plausible on its own.
//
// The third record this cell used to pin - the memory model's `lean` flag
// (inside the model's audit block) - was deleted with the model
// on 2026-09-17. The member name is now the record's only answer, which is
// what the record's own note said it was for.
TEST(SelectionReportingTest, BuilderMemberAgreesWithTheFamilyWord) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // The three request shapes reachable at this size: the absent key (the
    // auto-lean default), the explicit within-family word, and the
    // explicit family word (which keeps the machinery - Ruling B).
    const std::pair<const char*, const char*> fixtures[] = {
        {kH2AutoToml, "lean"}, {kH2LeanToml, "lean"}, {kH2DirectToml, "direct"}};

    for (const auto& [toml, expectedMember] : fixtures)
    {
        auto result = RunInputText(toml);
        ASSERT_TRUE(result.has_value()) << result.error().message;

        const auto selection = SelectionPos(*result);
        ASSERT_NE(selection, std::string::npos);

        const auto family = JsonString(*result, "\"builder\":", selection);
        const auto member = JsonString(*result, "\"builder_member\":", selection);
        ASSERT_TRUE(family.has_value());
        ASSERT_TRUE(member.has_value()) << "every resolved run must name the member it wired";

        EXPECT_EQ(*member, expectedMember);

        if (*member == "lean")
        {
            // The lean member is the direct family's within-family member:
            // the family word stays "direct".
            EXPECT_EQ(*family, "direct");
        } else
        {
            // Everywhere else the member IS the family's own builder.
            EXPECT_EQ(*member, *family);
        }

        // The lean spelling names no family word, so explicit_builder stays
        // absent by design - the honest absence the member name replaces as
        // the record's answer (never infer a value the record does not carry).
        if (*member == "lean")
        {
            EXPECT_EQ(JsonString(*result, "\"explicit_builder\":", selection), std::nullopt);
        }
    }
}
