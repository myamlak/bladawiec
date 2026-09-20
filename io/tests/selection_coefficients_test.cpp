// The coefficient-file loader acceptance (selection_coefficients.hpp):
// the crafted-file battery of the engine seam's own record - schema-
// version pins (older AND newer rejected whole), the strict key-set and
// token contract, the fit-metadata validation (condition-number ceiling,
// interpolation fail-closed, the covariance symmetric/PSD/dims), the
// per-cell exactly-one-payload rule, the drift-token vocabulary, the
// duplicate/overlap refusals, the deterministic canonical sort, and the
// superset tolerance of unknown cells. Every file here is crafted - the
// loader is data-independent by construction (no committed coefficient
// file exists).

#include "qcx/io/selection_coefficients.hpp"

#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

namespace {

using qcx::io::CanonicalCellKey;
using qcx::io::CoefficientCell;
using qcx::io::CoefficientDrift;
using qcx::io::ParseSelectionCoefficients;
using qcx::io::SelectionCoefficients;

// The canonical valid fit-metadata block: well-conditioned, rank 1, no
// aliasing, the 1x1 identity covariance, interpolation off.
constexpr char kFitMetadata[] = R"("fit_metadata": {"condition_number": 1.0,
                                                    "active_rank": 1,
                                                    "aliased_directions": [],
                                                    "covariance": [[1.0]],
                                                    "interpolation_enabled": false})";

// A measured cell over the fixture key (machine "m1", basis "ccpvdz",
// size band [100:199], preset "default", threads [1:16]).
std::string MeasuredCell(const std::string& family,
                         double median,
                         double mad = 0.0,
                         const std::string& drift = "normal",
                         int staleness = 0,
                         int nMeasurements = 5,
                         const std::string& key = "") {
    const std::string keyPart =
        key.empty()
            ? R"("key": {"path_family": ")" + family +
                  R"(", "machine_class_key": "m1", "basis_family_id": "ccpvdz", )" +
                  R"("n_basis_lower": 100, "n_basis_upper": 199, "preset_id": "default", )" +
                  R"("threads_lower": 1, "threads_upper": 16})"
            : R"("key": )" + key;

    return "{" + keyPart + ", \"median\": " + std::to_string(median) +
           ", \"mad\": " + std::to_string(mad) +
           ", \"n_measurements\": " + std::to_string(nMeasurements) +
           ", \"staleness_version\": " + std::to_string(staleness) + ", \"drift_state\": \"" +
           drift + "\"}";
}

// A borrowed cell over the fixture key.
std::string BorrowedCell(const std::string& family,
                         const std::string& donorFamily,
                         double inflation,
                         const std::string& drift = "normal") {
    return R"({"key": {"path_family": ")" + family +
           R"(", "machine_class_key": "m1", "basis_family_id": "ccpvdz", )" +
           R"("n_basis_lower": 100, "n_basis_upper": 199, "preset_id": "default", )" +
           R"("threads_lower": 1, "threads_upper": 16}, "borrowed": {"donor_family": ")" +
           donorFamily + R"(", "inflation": )" + std::to_string(inflation) +
           R"(}, "staleness_version": 0, "drift_state": ")" + drift + "\"}";
}

// Wraps cell JSON fragments into a whole file with the canonical valid
// fit metadata.
std::string MakeFile(const std::string& cells, const std::string& fileVersion = "v1") {
    return R"({"schema_version": 1, "file_version": ")" + fileVersion + "\", " + kFitMetadata +
           ", \"cells\": [" + cells + "]}";
}

// Loads and asserts success; returns the model (a failed load returns the
// default model and the assertions above already failed loudly).
SelectionCoefficients Load(const std::string& json) {
    const auto result = ParseSelectionCoefficients(json);
    EXPECT_TRUE(result.has_value());
    return result.has_value() ? *result : SelectionCoefficients{};
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void ExpectRejected(const std::string& json, const std::string& messagePart) {
    const auto result = ParseSelectionCoefficients(json);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find(messagePart), std::string::npos)
        << "message: " << result.error().message << "\njson: " << json;
}

TEST(SelectionCoefficientsTest, SchemaVersionExactMatchOnly) {
    // The schema-version pin: older AND newer files are rejected
    // whole (fail-closed) - a version-mismatched table must never
    // half-apply. The rejection names both versions.
    std::string file = MakeFile(MeasuredCell("ri_j", 1.0));
    file.replace(file.find(R"("schema_version": 1)"),
                 std::string(R"("schema_version": 1)").size(),
                 R"("schema_version": 0)");
    ExpectRejected(file, "schema_version 0 is not readable (this build reads exactly 1)");

    file = MakeFile(MeasuredCell("ri_j", 1.0));
    file.replace(file.find(R"("schema_version": 1)"),
                 std::string(R"("schema_version": 1)").size(),
                 R"("schema_version": 2)");
    ExpectRejected(file, "schema_version 2 is not readable (this build reads exactly 1)");

    EXPECT_EQ(qcx::io::kSelectionCoefficientsSchemaVersion, 1);
}

TEST(SelectionCoefficientsTest, MeasuredCellsLoadWithTheirVerdicts) {
    const SelectionCoefficients coefficients =
        Load(MakeFile(MeasuredCell("ri_j", 0.9, 0.05, "stale", 3, 6) + ", " +
                      MeasuredCell("direct_screened", 4.2)));

    ASSERT_EQ(coefficients.cells.size(), 2u);
    EXPECT_EQ(coefficients.fileVersion, "v1");
    EXPECT_EQ(coefficients.schemaVersion, 1);

    // The cells load canonical-key-sorted ("direct_screened@..." before
    // "ri_j@..." whatever the file order).
    const CoefficientCell& direct = coefficients.cells[0];
    EXPECT_EQ(direct.key.pathFamily, "direct_screened");
    EXPECT_EQ(direct.key.machineClassKey, "m1");
    EXPECT_EQ(direct.key.basisFamilyId, "ccpvdz");
    EXPECT_EQ(direct.key.nBasisLower, 100);
    EXPECT_EQ(direct.key.nBasisUpper, 199);
    EXPECT_EQ(direct.key.presetId, "default");
    EXPECT_EQ(direct.key.threadsLower, 1);
    EXPECT_EQ(direct.key.threadsUpper, 16);
    ASSERT_TRUE(direct.median.has_value());
    EXPECT_EQ(*direct.median, 4.2);
    EXPECT_EQ(*direct.mad, 0.0);
    EXPECT_EQ(direct.nMeasurements, 5);
    EXPECT_EQ(direct.stalenessVersion, 0);
    EXPECT_EQ(direct.drift, CoefficientDrift::kNormal);
    EXPECT_FALSE(direct.borrow.has_value());

    const CoefficientCell& riJ = coefficients.cells[1];
    EXPECT_EQ(riJ.key.pathFamily, "ri_j");
    EXPECT_EQ(*riJ.median, 0.9);
    EXPECT_EQ(*riJ.mad, 0.05);
    EXPECT_EQ(riJ.nMeasurements, 6);
    EXPECT_EQ(riJ.stalenessVersion, 3);
    EXPECT_EQ(riJ.drift, CoefficientDrift::kStale);
}

TEST(SelectionCoefficientsTest, FrozenDriftLoadsAsFrozen) {
    const SelectionCoefficients coefficients =
        Load(MakeFile(MeasuredCell("ri_j", 1.0, 0.0, "frozen", 9)));
    ASSERT_EQ(coefficients.cells.size(), 1u);
    EXPECT_EQ(coefficients.cells[0].drift, CoefficientDrift::kFrozen);
    EXPECT_EQ(coefficients.cells[0].stalenessVersion, 9);
}

TEST(SelectionCoefficientsTest, BorrowedCellLoads) {
    const SelectionCoefficients coefficients =
        Load(MakeFile(BorrowedCell("gpu", "direct_screened", 0.25, "stale")));
    ASSERT_EQ(coefficients.cells.size(), 1u);

    const CoefficientCell& gpu = coefficients.cells[0];
    EXPECT_EQ(gpu.key.pathFamily, "gpu");
    EXPECT_FALSE(gpu.median.has_value());
    EXPECT_EQ(gpu.nMeasurements, 0);
    EXPECT_EQ(gpu.drift, CoefficientDrift::kStale);
    ASSERT_TRUE(gpu.borrow.has_value());
    EXPECT_EQ(gpu.borrow->donorFamily, "direct_screened");
    EXPECT_EQ(gpu.borrow->inflation, 0.25);
}

TEST(SelectionCoefficientsTest, EmptyCellTableIsValid) {
    const SelectionCoefficients coefficients = Load(MakeFile(""));
    EXPECT_TRUE(coefficients.cells.empty());
    EXPECT_EQ(coefficients.fileVersion, "v1");
}

TEST(SelectionCoefficientsTest, FileLoadMatchesStringParse) {
    const std::string path = std::string(::testing::TempDir()) + "qcx-coefficients-test.json";
    const std::string content =
        MakeFile(MeasuredCell("qfmm", 3.0, 0.1) + ", " + MeasuredCell("ri_j", 1.0));
    {
        std::ofstream stream(path);
        stream << content;
    }

    const auto loaded = qcx::io::LoadSelectionCoefficientsFile(path);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->fileVersion, "v1");
    ASSERT_EQ(loaded->cells.size(), 2u);
    EXPECT_EQ(loaded->cells[0].key.pathFamily, "qfmm");
    EXPECT_EQ(loaded->cells[1].key.pathFamily, "ri_j");
}

TEST(SelectionCoefficientsTest, MissingFileIsAnIOError) {
    const auto result = qcx::io::LoadSelectionCoefficientsFile(std::string(::testing::TempDir()) +
                                                               "qcx-no-such-coefficient-file.json");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kIOError);
    EXPECT_NE(result.error().message.find("cannot open for reading"), std::string::npos);
}

TEST(SelectionCoefficientsTest, NotJsonIsRejected) {
    ExpectRejected("not json at all", "not readable JSON");
}

TEST(SelectionCoefficientsTest, UnknownAndMissingRootKeysAreRejected) {
    std::string file = MakeFile("");
    file.replace(
        file.find("\"cells\""), std::string("\"cells\"").size(), "\"extra\": 1, \"cells\"");
    ExpectRejected(file, "unknown key \"extra\" in the coefficient-file root");

    ExpectRejected(R"({"schema_version": 1, "file_version": "v1", "cells": []})",
                   "missing required key \"fit_metadata\" in the coefficient-file root");
}

TEST(SelectionCoefficientsTest, FileVersionMustBeACorpusToken) {
    ExpectRejected(MakeFile("", "NoT-a-token"), "is not a valid token");
    ExpectRejected(MakeFile("", ""), "is not a valid token");
}

TEST(SelectionCoefficientsTest, NonTokenKeyComponentsAreRejected) {
    // The corpus token charset ([a-z0-9][a-z0-9._-]*): an uppercase or
    // space-carrying machine key is not a publishable id.
    const std::string key = R"({"path_family": "ri_j", "machine_class_key": "M1", )"
                            R"("basis_family_id": "ccpvdz", "n_basis_lower": 100, )"
                            R"("n_basis_upper": 199, "preset_id": "default", )"
                            R"("threads_lower": 1, "threads_upper": 16})";
    ExpectRejected(MakeFile(MeasuredCell("ri_j", 1.0, 0.0, "normal", 0, 5, key)),
                   "\"machine_class_key\" = \"M1\"");
}

TEST(SelectionCoefficientsTest, UnknownPathFamilyIsRejectedWhole) {
    ExpectRejected(MakeFile(MeasuredCell("gpu_split", 1.0)),
                   "path_family \"gpu_split\" is not one of the ladder families");
}

TEST(SelectionCoefficientsTest, CellKeyContractIsStrict) {
    // An unknown key in a cell key (a schema typo like "n_basis") must not
    // silently drop: the whole file is rejected.
    std::string cell = MeasuredCell("ri_j", 1.0);
    cell.replace(cell.find("\"n_basis_lower\""),
                 std::string("\"n_basis_lower\"").size(),
                 "\"n_basis\": 100, \"n_basis_lower\"");
    ExpectRejected(MakeFile(cell), "unknown key \"n_basis\" in a cell key");
}

TEST(SelectionCoefficientsTest, InvertedAndZeroBandsAreRejected) {
    std::string cell = MeasuredCell("ri_j", 1.0);
    cell.replace(cell.find("\"n_basis_upper\": 199"),
                 std::string("\"n_basis_upper\": 199").size(),
                 "\"n_basis_upper\": 99");
    ExpectRejected(MakeFile(cell), "inverted size band");

    ExpectRejected(
        MakeFile(MeasuredCell(
            "ri_j",
            1.0,
            0.0,
            "normal",
            0,
            5,
            R"({"path_family": "ri_j", "machine_class_key": "m1", "basis_family_id": "ccpvdz", "n_basis_lower": 0, "n_basis_upper": 199, "preset_id": "default", "threads_lower": 1, "threads_upper": 16})")),
        "\"n_basis_lower\" = 0");
}

TEST(SelectionCoefficientsTest, ExactlyOnePayloadPerCell) {
    // Both payloads at once - rejected.
    std::string cell = MeasuredCell("gpu", 1.0);
    cell.replace(cell.find("\"median\""),
                 std::string("\"median\"").size(),
                 R"("borrowed": {"donor_family": "ri_j", "inflation": 1.5}, "median")");
    ExpectRejected(MakeFile(cell), "must carry exactly one payload");

    // Neither payload (a bare key) - rejected.
    ExpectRejected(
        MakeFile(
            MeasuredCell("ri_j", 1.0).substr(0, 0) +
            R"({"key": {"path_family": "ri_j", "machine_class_key": "m1", "basis_family_id": "ccpvdz", "n_basis_lower": 100, "n_basis_upper": 199, "preset_id": "default", "threads_lower": 1, "threads_upper": 16}, "staleness_version": 0, "drift_state": "normal"})"),
        "must carry exactly one payload");

    // A partial measured payload (median only) - rejected, never half-read.
    ExpectRejected(
        MakeFile(
            MeasuredCell("ri_j", 1.0).substr(0, 0) +
            R"({"key": {"path_family": "ri_j", "machine_class_key": "m1", "basis_family_id": "ccpvdz", "n_basis_lower": 100, "n_basis_upper": 199, "preset_id": "default", "threads_lower": 1, "threads_upper": 16}, "median": 1.0, "staleness_version": 0, "drift_state": "normal"})"),
        "must carry exactly one payload");
}

TEST(SelectionCoefficientsTest, BorrowContractIsStrict) {
    // A borrowed cell that borrows from its own family is nonsense.
    ExpectRejected(MakeFile(BorrowedCell("ri_j", "ri_j", 1.5)),
                   "borrows from its own family \"ri_j\"");

    // A donor outside the ladder set is rejected whole.
    ExpectRejected(MakeFile(BorrowedCell("gpu", "mp2", 1.5)),
                   "donor_family \"mp2\" is not one of the ladder families");

    // The inflation must be a positive finite number.
    ExpectRejected(MakeFile(BorrowedCell("gpu", "ri_j", 0.0)), "must be > 0");
    ExpectRejected(MakeFile(BorrowedCell("gpu", "ri_j", -1.0)), "must be > 0");
}

TEST(SelectionCoefficientsTest, MeasuredNumericContractIsStrict) {
    // A zero median is nonsense (no measured work is free) and a string
    // median is a type violation - both reject the whole file.
    std::string zero = MeasuredCell("ri_j", 1.0);
    zero.replace(zero.find("\"median\": 1.000000"),
                 std::string("\"median\": 1.000000").size(),
                 "\"median\": 0.0");
    ExpectRejected(MakeFile(zero), "\"median\" must be > 0");

    std::string asString = MeasuredCell("ri_j", 1.0);
    asString.replace(asString.find("\"median\": 1.000000"),
                     std::string("\"median\": 1.000000").size(),
                     "\"median\": \"1.0\"");
    ExpectRejected(MakeFile(asString),
                   "\"median\" in cell ri_j@m1@ccpvdz[100:199]@default@t[1:16] must be "
                   "a number");

    std::string noMeasurements = MeasuredCell("ri_j", 1.0);
    noMeasurements.replace(noMeasurements.find("\"n_measurements\": 5"),
                           std::string("\"n_measurements\": 5").size(),
                           "\"n_measurements\": 0");
    ExpectRejected(MakeFile(noMeasurements), "\"n_measurements\" = 0");
}

TEST(SelectionCoefficientsTest, VerdictFieldsAreStrict) {
    ExpectRejected(MakeFile(MeasuredCell("ri_j", 1.0, 0.0, "drifting")),
                   "\"drift_state\" = \"drifting\" in cell "
                   "ri_j@m1@ccpvdz[100:199]@default@t[1:16] is not one of "
                   "normal/stale/frozen");

    std::string negative = MeasuredCell("ri_j", 1.0);
    negative.replace(negative.find("\"staleness_version\": 0"),
                     std::string("\"staleness_version\": 0").size(),
                     "\"staleness_version\": -1");
    ExpectRejected(MakeFile(negative), "\"staleness_version\" = -1");
}

TEST(SelectionCoefficientsTest, ConditionNumberCeilingIsEnforced) {
    // The "~1e6" hard threshold: an ill-conditioned fit is rejected
    // whole. 1e6 itself is readable; 1e6 + 1 is not.
    const std::string file =
        R"({"schema_version": 1, "file_version": "v1", "fit_metadata": {"condition_number": 1000001.0, "active_rank": 1, "aliased_directions": [], "covariance": [[1.0]], "interpolation_enabled": false}, "cells": []})";
    ExpectRejected(file, "condition_number");

    const std::string ok =
        R"({"schema_version": 1, "file_version": "v1", "fit_metadata": {"condition_number": 1000000.0, "active_rank": 1, "aliased_directions": [], "covariance": [[1.0]], "interpolation_enabled": false}, "cells": []})";
    EXPECT_TRUE(ParseSelectionCoefficients(ok).has_value());
}

TEST(SelectionCoefficientsTest, SubUnitConditionNumberIsRejected) {
    std::string file =
        R"({"schema_version": 1, "file_version": "v1", "fit_metadata": {"condition_number": 0.5, "active_rank": 1, "aliased_directions": [], "covariance": [[1.0]], "interpolation_enabled": false}, "cells": []})";
    ExpectRejected(file, "\"condition_number\" = 0.500000");
}

TEST(SelectionCoefficientsTest, ActiveRankMustBePositive) {
    std::string file =
        R"({"schema_version": 1, "file_version": "v1", "fit_metadata": {"condition_number": 1.0, "active_rank": 0, "aliased_directions": [], "covariance": [[1.0]], "interpolation_enabled": false}, "cells": []})";
    ExpectRejected(file, "\"active_rank\" = 0");
}

TEST(SelectionCoefficientsTest, InterpolationEnabledIsFailClosed) {
    // The fallback chain's same-family interpolation is NOT implemented in
    // the engine seam: a file that enables it is rejected whole rather
    // than silently under-applied (a fit that intends interpolation must
    // not have its raw cells read as the intended prices).
    std::string file =
        R"({"schema_version": 1, "file_version": "v1", "fit_metadata": {"condition_number": 1.0, "active_rank": 1, "aliased_directions": [], "covariance": [[1.0]], "interpolation_enabled": true}, "cells": []})";
    ExpectRejected(file, "interpolation_enabled = true is not implemented in the engine seam");
}

TEST(SelectionCoefficientsTest, CovarianceDimsSymmetryAndPsdAreValidated) {
    // Not square.
    std::string file =
        R"({"schema_version": 1, "file_version": "v1", "fit_metadata": {"condition_number": 1.0, "active_rank": 1, "aliased_directions": [], "covariance": [[1.0, 0.0]], "interpolation_enabled": false}, "cells": []})";
    ExpectRejected(file, "must be a square matrix");

    // Not symmetric.
    file =
        R"({"schema_version": 1, "file_version": "v1", "fit_metadata": {"condition_number": 1.0, "active_rank": 1, "aliased_directions": [], "covariance": [[1.0, 0.5], [0.4, 1.0]], "interpolation_enabled": false}, "cells": []})";
    ExpectRejected(file, "not symmetric");

    // Indefinite.
    file =
        R"({"schema_version": 1, "file_version": "v1", "fit_metadata": {"condition_number": 1.0, "active_rank": 1, "aliased_directions": [], "covariance": [[1.0, 2.0], [2.0, 1.0]], "interpolation_enabled": false}, "cells": []})";
    ExpectRejected(file, "not positive semidefinite");

    // Semidefinite is a valid covariance (eigenvalues 2 and 0).
    file =
        R"({"schema_version": 1, "file_version": "v1", "fit_metadata": {"condition_number": 1.0, "active_rank": 1, "aliased_directions": [], "covariance": [[1.0, 1.0], [1.0, 1.0]], "interpolation_enabled": false}, "cells": []})";
    EXPECT_TRUE(ParseSelectionCoefficients(file).has_value());
}

TEST(SelectionCoefficientsTest, AliasedDirectionsAreTokensWithoutDuplicates) {
    std::string file =
        R"({"schema_version": 1, "file_version": "v1", "fit_metadata": {"condition_number": 1.0, "active_rank": 1, "aliased_directions": ["a1", "a1"], "covariance": [[1.0]], "interpolation_enabled": false}, "cells": []})";
    ExpectRejected(file, "aliased_directions lists \"a1\" twice");

    file =
        R"({"schema_version": 1, "file_version": "v1", "fit_metadata": {"condition_number": 1.0, "active_rank": 1, "aliased_directions": ["Not a token"], "covariance": [[1.0]], "interpolation_enabled": false}, "cells": []})";
    ExpectRejected(file, "is not a valid token");
}

TEST(SelectionCoefficientsTest, DuplicateCellsAreRejected) {
    const std::string duplicate =
        MakeFile(MeasuredCell("ri_j", 1.0) + ", " + MeasuredCell("ri_j", 2.0));
    ExpectRejected(duplicate,
                   "two cells share the canonical key ri_j@m1@ccpvdz[100:199]@default@t[1:16]");
}

TEST(SelectionCoefficientsTest, CoverageOverlapIsRejected) {
    // Two ri_j cells of the same machine/basis/preset whose size bands
    // intersect at the same thread band would make a lookup ambiguous.
    std::string overlapping = MakeFile(
        MeasuredCell("ri_j", 1.0) + ", " +
        MeasuredCell(
            "ri_j",
            2.0,
            0.0,
            "normal",
            0,
            5,
            R"({"path_family": "ri_j", "machine_class_key": "m1", "basis_family_id": "ccpvdz", "n_basis_lower": 150, "n_basis_upper": 249, "preset_id": "default", "threads_lower": 1, "threads_upper": 16})"));
    ExpectRejected(overlapping, "overlap in coverage");
}

TEST(SelectionCoefficientsTest, AdjacentBandsAndDisjointThreadBandsAreFine) {
    // [100:199] next to [200:299] of the same family/machine/basis/preset
    // is unambiguous, and so are same-size cells whose thread bands are
    // disjoint - the coverage rule needs BOTH bands to intersect.
    const std::string adjacent = MakeFile(
        MeasuredCell("ri_j", 1.0) + ", " +
        MeasuredCell(
            "ri_j",
            2.0,
            0.0,
            "normal",
            0,
            5,
            R"({"path_family": "ri_j", "machine_class_key": "m1", "basis_family_id": "ccpvdz", "n_basis_lower": 200, "n_basis_upper": 299, "preset_id": "default", "threads_lower": 1, "threads_upper": 16})") +
        ", " +
        MeasuredCell(
            "ri_j",
            3.0,
            0.0,
            "normal",
            0,
            5,
            R"({"path_family": "ri_j", "machine_class_key": "m1", "basis_family_id": "ccpvdz", "n_basis_lower": 100, "n_basis_upper": 199, "preset_id": "default", "threads_lower": 17, "threads_upper": 32})"));
    const SelectionCoefficients coefficients = Load(adjacent);
    ASSERT_EQ(coefficients.cells.size(), 3u);
}

TEST(SelectionCoefficientsTest, UnknownCellsAreToleratedAtLoad) {
    // The superset tolerance: cells for other machines, basis families,
    // presets and sizes are unknown to a run's lookups but never poison
    // the file - load succeeds with every cell present.
    const std::string superset = MakeFile(
        MeasuredCell("ri_j", 1.0) + ", " +
        MeasuredCell(
            "gpu",
            0.2,
            0.0,
            "normal",
            0,
            5,
            R"({"path_family": "gpu", "machine_class_key": "other-machine", "basis_family_id": "def2svp", "n_basis_lower": 300, "n_basis_upper": 599, "preset_id": "tight", "threads_lower": 1, "threads_upper": 8})") +
        ", " + MeasuredCell("qfmm", 5.0));
    const SelectionCoefficients coefficients = Load(superset);
    ASSERT_EQ(coefficients.cells.size(), 3u);
}

TEST(SelectionCoefficientsTest, CanonicalCellKeyFormat) {
    // The canonical key string of a loaded cell: the format the file
    // contract and the run record share ("{family}@{machine}@{basis}
    // [{lo}:{hi}]@{preset}@t[{lo}:{hi}]"). The model outlives the reads -
    // no reference into a temporary.
    const SelectionCoefficients model = Load(MakeFile(MeasuredCell("ri_j", 1.0)));
    ASSERT_EQ(model.cells.size(), 1u);
    EXPECT_EQ(CanonicalCellKey(model.cells[0].key), "ri_j@m1@ccpvdz[100:199]@default@t[1:16]");
}

TEST(SelectionCoefficientsTest, FindCoefficientCellMatchesCoverageOnly) {
    const SelectionCoefficients coefficients = Load(MakeFile(
        MeasuredCell("ri_j", 1.0) + ", " +
        MeasuredCell(
            "ri_j",
            9.0,
            0.0,
            "normal",
            0,
            5,
            R"({"path_family": "ri_j", "machine_class_key": "m1", "basis_family_id": "ccpvdz", "n_basis_lower": 200, "n_basis_upper": 299, "preset_id": "default", "threads_lower": 1, "threads_upper": 16})")));

    using qcx::io::FindCoefficientCell;

    // 150 basis functions at 8 threads hit the [100:199] cell.
    const CoefficientCell* found =
        FindCoefficientCell(coefficients, "ri_j", "m1", "ccpvdz", "default", 150, 8);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(*found->median, 1.0);

    // Band edges are inclusive.
    EXPECT_NE(FindCoefficientCell(coefficients, "ri_j", "m1", "ccpvdz", "default", 100, 16),
              nullptr);
    EXPECT_NE(FindCoefficientCell(coefficients, "ri_j", "m1", "ccpvdz", "default", 199, 1),
              nullptr);

    // 250 basis functions at 8 threads hit the [200:299] cell.
    const CoefficientCell* upper =
        FindCoefficientCell(coefficients, "ri_j", "m1", "ccpvdz", "default", 200, 8);
    ASSERT_NE(upper, nullptr);
    EXPECT_EQ(*upper->median, 9.0);

    // Out-of-band, other family, other machine and other preset miss.
    EXPECT_EQ(FindCoefficientCell(coefficients, "ri_j", "m1", "ccpvdz", "default", 300, 8),
              nullptr);
    EXPECT_EQ(FindCoefficientCell(coefficients, "gpu", "m1", "ccpvdz", "default", 150, 8), nullptr);
    EXPECT_EQ(FindCoefficientCell(coefficients, "ri_j", "m2", "ccpvdz", "default", 150, 8),
              nullptr);
    EXPECT_EQ(FindCoefficientCell(coefficients, "ri_j", "m1", "ccpvdz", "loose", 150, 8), nullptr);
}

TEST(SelectionCoefficientsTest, LoadTimeOfAManyCellFileIsNegligible) {
    // The load-time-overhead pin: the one-time per-run load+validation of
    // a full-size crafted file (250 cells across every family) completes
    // in well under a second - negligible against the multi-minute HF run
    // it precedes (no timing assertions tighter than that: this is a
    // functional floor, not a benchmark).
    std::string cells;

    for (int machine = 0; machine < 5; ++machine)
    {
        for (int family = 0; family < 5; ++family)
        {
            for (int band = 0; band < 10; ++band)
            {
                if (!cells.empty())
                {
                    cells += ", ";
                }

                cells += R"({"key": {"path_family": ")" +
                         std::string(qcx::io::kCoefficientPathFamilies[family]) +
                         R"(", "machine_class_key": "bench-m)" + std::to_string(machine) +
                         R"(", "basis_family_id": "ccpvdz", "n_basis_lower": )" +
                         std::to_string(100 * band + 1) +
                         ", \"n_basis_upper\": " + std::to_string(100 * (band + 1)) +
                         R"(, "preset_id": "default", "threads_lower": 1, "threads_upper": 32}, )" +
                         R"("median": 1.0, "mad": 0.1, "n_measurements": 4, )" +
                         R"("staleness_version": 0, "drift_state": "normal"})";
            }
        }
    }

    const std::string file = MakeFile(cells);
    const auto start = std::chrono::steady_clock::now();
    const auto result = ParseSelectionCoefficients(file);
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();

    ASSERT_TRUE(result.has_value()) << "load rejected the full-size file: "
                                    << (result.has_value() ? "" : result.error().message);
    EXPECT_EQ(result->cells.size(), 250u);
    EXPECT_LT(elapsedMs, 1000);
}

} // namespace
