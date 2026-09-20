// The engine-side selection-coefficient file loader
// (selection_coefficients.hpp): the strict load-time validation of the
// published measured-cost table - every schema-version, key-set, token,
// numeric, payload and coverage invariant of the file contract rejects the
// WHOLE file (fail-closed), so a rejected table behaves exactly like no
// table. The seam policy that consumes the loaded model lives in the
// driver (selection_heuristic.cpp); this module only parses and validates.

#include "qcx/io/selection_coefficients.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>

namespace qcx::io {
namespace {

// The symmetry/PSD tolerance of the covariance validation, relative to the
// matrix's magnitude scale (max(1, max |entry|)): publishers round-trip
// through JSON at ~17 significant digits, so an exact PSD certificate is
// not the bar - a published matrix that is symmetric and PSD up to
// round-off is a valid covariance.
constexpr double kCovarianceTolerance = 1e-9;

qcx::Error Reject(std::string message) {
    return qcx::Error{qcx::ErrorCode::kInvalidArgument, std::move(message)};
}

// The corpus token charset ([a-z0-9][a-z0-9._-]*,
// tools/bench/cost_table_schema.py): the ids a coefficient file may carry
// (machine-class keys, basis-family ids, preset ids, version ids, aliased
// directions).
bool IsCorpusToken(std::string_view text) {
    if (text.empty())
    {
        return false;
    }

    const auto isFirst = [](char c) { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'); };
    const auto isRest = [isFirst](char c) {
        return isFirst(c) || c == '.' || c == '_' || c == '-';
    };

    if (!isFirst(text.front()))
    {
        return false;
    }

    return std::all_of(text.begin() + 1, text.end(), isRest);
}

// The context string of a message ("coefficient file: <detail>") is
// spelled out at every site so each rejection reads standalone.

// Checks that a node is an object and that its key set is exactly the
// known keys with the required ones present.
template <std::size_t kKnown, std::size_t kRequired>
std::optional<qcx::Error> CheckObjectKeys(const nlohmann::json& node,
                                          std::string_view context,
                                          const std::array<std::string_view, kKnown>& known,
                                          const std::array<std::string_view, kRequired>& required) {
    if (!node.is_object())
    {
        return Reject(std::string(context) + " must be a JSON object");
    }

    for (auto it = node.begin(); it != node.end(); ++it)
    {
        if (std::find(known.begin(), known.end(), it.key()) == known.end())
        {
            return Reject("coefficient file: unknown key \"" + it.key() + "\" in " +
                          std::string(context));
        }
    }

    for (const std::string_view name : required)
    {
        if (!node.contains(std::string(name)))
        {
            return Reject("coefficient file: missing required key \"" + std::string(name) +
                          "\" in " + std::string(context));
        }
    }

    return std::nullopt;
}

// Reads a corpus-token string field.
std::optional<qcx::Error> ReadTokenField(const nlohmann::json& object,
                                         std::string_view name,
                                         std::string_view context,
                                         std::string& out) {
    const nlohmann::json& value = object[std::string(name)];

    if (!value.is_string())
    {
        return Reject("coefficient file: \"" + std::string(name) + "\" in " + std::string(context) +
                      " must be a string");
    }

    const std::string text = value.get<std::string>();

    if (!IsCorpusToken(text))
    {
        return Reject("coefficient file: \"" + std::string(name) + "\" = \"" + text + "\" in " +
                      std::string(context) + " is not a valid token ([a-z0-9][a-z0-9._-]*)");
    }

    out = text;
    return std::nullopt;
}

// Reads an integer field inside [min, max].
std::optional<qcx::Error> ReadIntField(const nlohmann::json& object,
                                       std::string_view name,
                                       std::string_view context,
                                       std::int64_t min,
                                       std::int64_t max,
                                       std::int64_t& out) {
    const nlohmann::json& value = object[std::string(name)];

    if (!value.is_number_integer())
    {
        return Reject("coefficient file: \"" + std::string(name) + "\" in " + std::string(context) +
                      " must be an integer");
    }

    const std::int64_t number = value.get<std::int64_t>();

    if (number < min || number > max)
    {
        return Reject("coefficient file: \"" + std::string(name) +
                      "\" = " + std::to_string(number) + " in " + std::string(context) +
                      " must be in [" + std::to_string(min) + ", " + std::to_string(max) + "]");
    }

    out = number;
    return std::nullopt;
}

// Reads a finite double field; optionally range-checked.
std::optional<qcx::Error> ReadDoubleField(const nlohmann::json& object,
                                          std::string_view name,
                                          std::string_view context,
                                          double min,
                                          double max,
                                          double& out) {
    const nlohmann::json& value = object[std::string(name)];

    if (!value.is_number())
    {
        return Reject("coefficient file: \"" + std::string(name) + "\" in " + std::string(context) +
                      " must be a number");
    }

    const double number = value.get<double>();

    if (!std::isfinite(number) || number < min || number > max)
    {
        return Reject("coefficient file: \"" + std::string(name) +
                      "\" = " + std::to_string(number) + " in " + std::string(context) +
                      " must be a finite number in [" + std::to_string(min) + ", " +
                      std::to_string(max) + "]");
    }

    out = number;
    return std::nullopt;
}

std::string_view DriftToken(CoefficientDrift drift) {
    switch (drift)
    {
    case CoefficientDrift::kNormal:
        return "normal";
    case CoefficientDrift::kStale:
        return "stale";
    case CoefficientDrift::kFrozen:
        return "frozen";
    }

    return "unknown";
}

bool IsLadderFamily(std::string_view family) {
    return std::find(kCoefficientPathFamilies.begin(), kCoefficientPathFamilies.end(), family) !=
           kCoefficientPathFamilies.end();
}

// Validates the covariance: nonempty square array of finite numbers,
// symmetric within the relative tolerance, positive semidefinite within it
// (the eigendecomposition runs on the symmetrized matrix). The load-time
// validation is the "validated - PSD, symmetric, dimensions" of the
// fit-metadata contract; the covariance is never used for selection.
std::optional<qcx::Error> ParseCovariance(const nlohmann::json& node,
                                          std::vector<std::vector<double>>& out) {
    constexpr std::string_view kContext = "fit_metadata.covariance";

    if (!node.is_array())
    {
        return Reject("coefficient file: " + std::string(kContext) + " must be an array");
    }

    if (node.empty())
    {
        return Reject("coefficient file: " + std::string(kContext) +
                      " must be nonempty (the fit has at least one parameter)");
    }

    const std::size_t dims = node.size();
    double maxAbs = 1.0;

    for (std::size_t i = 0; i < dims; ++i)
    {
        const nlohmann::json& row = node[i];

        if (!row.is_array() || row.size() != dims)
        {
            return Reject("coefficient file: " + std::string(kContext) +
                          " must be a square matrix (row " + std::to_string(i) + " has " +
                          std::to_string(row.is_array() ? row.size() : 0) + " entries, expected " +
                          std::to_string(dims) + ")");
        }

        std::vector<double> values(dims);

        for (std::size_t j = 0; j < dims; ++j)
        {
            if (!row[j].is_number())
            {
                return Reject("coefficient file: " + std::string(kContext) + " entry [" +
                              std::to_string(i) + "][" + std::to_string(j) + "] must be a number");
            }

            values[j] = row[j].get<double>();

            if (!std::isfinite(values[j]))
            {
                return Reject("coefficient file: " + std::string(kContext) + " entry [" +
                              std::to_string(i) + "][" + std::to_string(j) + "] must be finite");
            }

            maxAbs = std::max(maxAbs, std::abs(values[j]));
        }

        out.push_back(std::move(values));
    }

    const double tolerance = kCovarianceTolerance * maxAbs;

    for (std::size_t i = 0; i < dims; ++i)
    {
        for (std::size_t j = 0; j < dims; ++j)
        {
            if (std::abs(out[i][j] - out[j][i]) > tolerance)
            {
                return Reject("coefficient file: fit_metadata.covariance is not symmetric "
                              "(entry [" +
                              std::to_string(i) + "][" + std::to_string(j) +
                              "] = " + std::to_string(out[i][j]) + " vs [" + std::to_string(j) +
                              "][" + std::to_string(i) + "] = " + std::to_string(out[j][i]) + ")");
            }
        }
    }

    Eigen::MatrixXd matrix(dims, dims);

    for (std::size_t i = 0; i < dims; ++i)
    {
        for (std::size_t j = 0; j < dims; ++j)
        {
            matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = out[i][j];
        }
    }

    const Eigen::MatrixXd symmetrized = 0.5 * (matrix + matrix.transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetrized);

    if (solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() < -tolerance)
    {
        return Reject("coefficient file: fit_metadata.covariance is not positive semidefinite "
                      "(minimum eigenvalue " +
                      std::to_string(solver.eigenvalues().minCoeff()) + ")");
    }

    return std::nullopt;
}

// One cell's key object, then the cell's payload and verdicts. Context
// messages name the cell by its canonical key once the key validates.
std::optional<qcx::Error> ParseCell(const nlohmann::json& node, CoefficientCell& cell) {
    constexpr std::string_view kContext = "a cell";

    if (auto error = CheckObjectKeys(
            node,
            kContext,
            std::to_array<std::string_view>({"key",
                                             "median",
                                             "mad",
                                             "n_measurements",
                                             "staleness_version",
                                             "drift_state",
                                             "borrowed"}),
            std::to_array<std::string_view>({"key", "staleness_version", "drift_state"}));
        error.has_value())
    {
        return error;
    }

    const nlohmann::json& keyNode = node["key"];

    if (auto error = CheckObjectKeys(keyNode,
                                     "a cell key",
                                     std::to_array<std::string_view>({"path_family",
                                                                      "machine_class_key",
                                                                      "basis_family_id",
                                                                      "n_basis_lower",
                                                                      "n_basis_upper",
                                                                      "preset_id",
                                                                      "threads_lower",
                                                                      "threads_upper"}),
                                     std::to_array<std::string_view>({"path_family",
                                                                      "machine_class_key",
                                                                      "basis_family_id",
                                                                      "n_basis_lower",
                                                                      "n_basis_upper",
                                                                      "preset_id",
                                                                      "threads_lower",
                                                                      "threads_upper"}));
        error.has_value())
    {
        return error;
    }

    CoefficientCellKey& key = cell.key;

    if (auto error = ReadTokenField(keyNode, "path_family", "a cell key", key.pathFamily);
        error.has_value())
    {
        return error;
    }

    if (!IsLadderFamily(key.pathFamily))
    {
        return Reject("coefficient file: a cell key's path_family \"" + key.pathFamily +
                      "\" is not one of the ladder families (direct_screened, light_path, ri_j, "
                      "qfmm, gpu)");
    }

    if (auto error =
            ReadTokenField(keyNode, "machine_class_key", "a cell key", key.machineClassKey);
        error.has_value())
    {
        return error;
    }

    if (auto error = ReadTokenField(keyNode, "basis_family_id", "a cell key", key.basisFamilyId);
        error.has_value())
    {
        return error;
    }

    std::int64_t band = 0;

    if (auto error = ReadIntField(keyNode, "n_basis_lower", "a cell key", 1, 1 << 30, band);
        error.has_value())
    {
        return error;
    }

    key.nBasisLower = static_cast<int>(band);

    if (auto error = ReadIntField(keyNode, "n_basis_upper", "a cell key", 1, 1 << 30, band);
        error.has_value())
    {
        return error;
    }

    key.nBasisUpper = static_cast<int>(band);

    if (auto error = ReadTokenField(keyNode, "preset_id", "a cell key", key.presetId);
        error.has_value())
    {
        return error;
    }

    if (auto error = ReadIntField(keyNode, "threads_lower", "a cell key", 1, 1 << 20, band);
        error.has_value())
    {
        return error;
    }

    key.threadsLower = static_cast<int>(band);

    if (auto error = ReadIntField(keyNode, "threads_upper", "a cell key", 1, 1 << 20, band);
        error.has_value())
    {
        return error;
    }

    key.threadsUpper = static_cast<int>(band);

    const std::string context = "cell " + CanonicalCellKey(key);

    if (key.nBasisUpper < key.nBasisLower)
    {
        return Reject("coefficient file: " + context +
                      " has an inverted size band (n_basis_upper < n_basis_lower)");
    }

    if (key.threadsUpper < key.threadsLower)
    {
        return Reject("coefficient file: " + context +
                      " has an inverted thread band (threads_upper < threads_lower)");
    }

    std::int64_t counter = 0;

    if (auto error = ReadIntField(node, "staleness_version", context, 0, 1 << 30, counter);
        error.has_value())
    {
        return error;
    }

    cell.stalenessVersion = static_cast<int>(counter);

    const nlohmann::json& driftNode = node["drift_state"];

    if (!driftNode.is_string())
    {
        return Reject("coefficient file: \"drift_state\" in " + context + " must be a string");
    }

    const std::string driftText = driftNode.get<std::string>();

    if (driftText == "normal")
    {
        cell.drift = CoefficientDrift::kNormal;
    } else if (driftText == "stale")
    {
        cell.drift = CoefficientDrift::kStale;
    } else if (driftText == "frozen")
    {
        cell.drift = CoefficientDrift::kFrozen;
    } else
    {
        return Reject("coefficient file: \"drift_state\" = \"" + driftText + "\" in " + context +
                      " is not one of normal/stale/frozen");
    }

    const bool hasBorrow = node.contains("borrowed");
    const bool hasMeasuredKey =
        node.contains("median") || node.contains("mad") || node.contains("n_measurements");
    const bool completeMeasured =
        node.contains("median") && node.contains("mad") && node.contains("n_measurements");

    // Exactly one payload: a borrowed cell carries no measured stats (they
    // would silently drop), and a measured cell is complete - a partial
    // measured payload is never half-read.
    if ((hasBorrow && hasMeasuredKey) || (!hasBorrow && !completeMeasured))
    {
        return Reject("coefficient file: " + context +
                      " must carry exactly one payload: the measured (median, mad, "
                      "n_measurements) or the borrowed (borrowed)");
    }

    if (hasBorrow)
    {
        const nlohmann::json& borrowNode = node["borrowed"];

        if (auto error =
                CheckObjectKeys(borrowNode,
                                "a borrowed payload",
                                std::to_array<std::string_view>({"donor_family", "inflation"}),
                                std::to_array<std::string_view>({"donor_family", "inflation"}));
            error.has_value())
        {
            return error;
        }

        CoefficientBorrow borrow;

        if (auto error = ReadTokenField(
                borrowNode, "donor_family", "a borrowed payload", borrow.donorFamily);
            error.has_value())
        {
            return error;
        }

        if (!IsLadderFamily(borrow.donorFamily))
        {
            return Reject("coefficient file: a borrowed payload's donor_family \"" +
                          borrow.donorFamily +
                          "\" is not one of the ladder families (direct_screened, light_path, "
                          "ri_j, qfmm, gpu)");
        }

        if (borrow.donorFamily == key.pathFamily)
        {
            return Reject("coefficient file: " + context + " borrows from its own family \"" +
                          borrow.donorFamily + "\"");
        }

        double inflation = 0.0;

        if (auto error = ReadDoubleField(
                borrowNode, "inflation", "a borrowed payload", -1e12, 1e12, inflation);
            error.has_value())
        {
            return error;
        }

        if (inflation <= 0.0)
        {
            return Reject("coefficient file: a borrowed payload's inflation must be > 0 in " +
                          context);
        }

        borrow.inflation = inflation;
        cell.borrow = std::move(borrow);
        return std::nullopt;
    }

    double value = 0.0;

    if (auto error = ReadDoubleField(node, "median", context, 0.0, 1e12, value); error.has_value())
    {
        return error;
    }

    if (value <= 0.0)
    {
        return Reject("coefficient file: \"median\" must be > 0 in " + context);
    }

    cell.median = value;

    if (auto error = ReadDoubleField(node, "mad", context, 0.0, 1e12, value); error.has_value())
    {
        return error;
    }

    cell.mad = value;

    if (auto error = ReadIntField(node, "n_measurements", context, 1, 1 << 30, counter);
        error.has_value())
    {
        return error;
    }

    cell.nMeasurements = static_cast<int>(counter);
    return std::nullopt;
}

} // namespace

std::string CanonicalCellKey(const CoefficientCellKey& key) {
    std::string result;
    result.reserve(96);
    result += key.pathFamily;
    result += '@';
    result += key.machineClassKey;
    result += '@';
    result += key.basisFamilyId;
    result += '[';
    result += std::to_string(key.nBasisLower);
    result += ':';
    result += std::to_string(key.nBasisUpper);
    result += "]@";
    result += key.presetId;
    result += "@t[";
    result += std::to_string(key.threadsLower);
    result += ':';
    result += std::to_string(key.threadsUpper);
    result += ']';
    return result;
}

const CoefficientCell* FindCoefficientCell(const SelectionCoefficients& coefficients,
                                           std::string_view pathFamily,
                                           std::string_view machineClassKey,
                                           std::string_view basisFamilyId,
                                           std::string_view presetId,
                                           // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                           std::size_t nBasis,
                                           int threads) noexcept {
    for (const CoefficientCell& cell : coefficients.cells)
    {
        const CoefficientCellKey& key = cell.key;

        if (key.pathFamily != pathFamily || key.machineClassKey != machineClassKey ||
            key.basisFamilyId != basisFamilyId || key.presetId != presetId)
        {
            continue;
        }

        const std::size_t n = static_cast<std::size_t>(key.nBasisLower);
        const std::size_t m = static_cast<std::size_t>(key.nBasisUpper);

        if (nBasis < n || nBasis > m)
        {
            continue;
        }

        const int lo = key.threadsLower;
        const int hi = key.threadsUpper;

        if (threads < lo || threads > hi)
        {
            continue;
        }

        return &cell;
    }

    return nullptr;
}

qcx::Result<SelectionCoefficients> ParseSelectionCoefficients(std::string_view json) noexcept {
    try
    {
        const nlohmann::json root = nlohmann::json::parse(json);

        if (auto error =
                CheckObjectKeys(root,
                                "the coefficient-file root",
                                std::to_array<std::string_view>(
                                    {"schema_version", "file_version", "fit_metadata", "cells"}),
                                std::to_array<std::string_view>(
                                    {"schema_version", "file_version", "fit_metadata", "cells"}));
            error.has_value())
        {
            return std::unexpected(*error);
        }

        const nlohmann::json& versionNode = root["schema_version"];

        if (!versionNode.is_number_integer())
        {
            return std::unexpected(
                Reject("coefficient file: \"schema_version\" must be an integer"));
        }

        const std::int64_t version = versionNode.get<std::int64_t>();

        if (version != kSelectionCoefficientsSchemaVersion)
        {
            return std::unexpected(Reject("coefficient file: schema_version " +
                                          std::to_string(version) +
                                          " is not readable (this build reads exactly " +
                                          std::to_string(kSelectionCoefficientsSchemaVersion) +
                                          "); the file is rejected whole (fail-closed)"));
        }

        SelectionCoefficients coefficients;
        coefficients.schemaVersion = static_cast<int>(version);

        if (auto error = ReadTokenField(
                root, "file_version", "the coefficient-file root", coefficients.fileVersion);
            error.has_value())
        {
            return std::unexpected(*error);
        }

        const nlohmann::json& fitNode = root["fit_metadata"];

        if (auto error =
                CheckObjectKeys(fitNode,
                                "fit_metadata",
                                std::to_array<std::string_view>({"condition_number",
                                                                 "active_rank",
                                                                 "aliased_directions",
                                                                 "covariance",
                                                                 "interpolation_enabled"}),
                                std::to_array<std::string_view>({"condition_number",
                                                                 "active_rank",
                                                                 "aliased_directions",
                                                                 "covariance",
                                                                 "interpolation_enabled"}));
            error.has_value())
        {
            return std::unexpected(*error);
        }

        CoefficientFitMetadata& fit = coefficients.fitMetadata;

        double condition = 0.0;

        if (auto error = ReadDoubleField(fitNode,
                                         "condition_number",
                                         "fit_metadata",
                                         1.0,
                                         kCoefficientMaxConditionNumber,
                                         condition);
            error.has_value())
        {
            return std::unexpected(*error);
        }

        fit.conditionNumber = condition;

        std::int64_t rank = 0;

        if (auto error = ReadIntField(fitNode, "active_rank", "fit_metadata", 1, 1 << 30, rank);
            error.has_value())
        {
            return std::unexpected(*error);
        }

        fit.activeRank = static_cast<int>(rank);

        const nlohmann::json& aliasesNode = fitNode["aliased_directions"];

        if (!aliasesNode.is_array())
        {
            return std::unexpected(
                Reject("coefficient file: \"aliased_directions\" in fit_metadata must be an "
                       "array"));
        }

        for (const nlohmann::json& alias : aliasesNode)
        {
            if (!alias.is_string())
            {
                return std::unexpected(
                    Reject("coefficient file: an aliased_directions entry must be a string"));
            }

            const std::string name = alias.get<std::string>();

            if (!IsCorpusToken(name))
            {
                return std::unexpected(Reject("coefficient file: aliased_directions entry \"" +
                                              name +
                                              "\" is not a valid token ([a-z0-9][a-z0-9._-]*)"));
            }

            if (std::find(fit.aliasedDirections.begin(), fit.aliasedDirections.end(), name) !=
                fit.aliasedDirections.end())
            {
                return std::unexpected(
                    Reject("coefficient file: aliased_directions lists \"" + name + "\" twice"));
            }

            fit.aliasedDirections.push_back(name);
        }

        if (auto error = ParseCovariance(fitNode["covariance"], fit.covariance); error.has_value())
        {
            return std::unexpected(*error);
        }

        const nlohmann::json& interpolationNode = fitNode["interpolation_enabled"];

        if (!interpolationNode.is_boolean())
        {
            return std::unexpected(
                Reject("coefficient file: \"interpolation_enabled\" in fit_metadata must be a "
                       "boolean"));
        }

        if (interpolationNode.get<bool>())
        {
            return std::unexpected(
                Reject("coefficient file: interpolation_enabled = true is not implemented in "
                       "the engine seam (schema 1 files never enable it; the file is rejected "
                       "whole, fail-closed)"));
        }

        const nlohmann::json& cellsNode = root["cells"];

        if (!cellsNode.is_array())
        {
            return std::unexpected(Reject("coefficient file: \"cells\" must be an array"));
        }

        for (const nlohmann::json& cellNode : cellsNode)
        {
            CoefficientCell cell;

            if (auto error = ParseCell(cellNode, cell); error.has_value())
            {
                return std::unexpected(*error);
            }

            coefficients.cells.push_back(std::move(cell));
        }

        // The pairwise disjoint-coverage validation: two cells that share
        // the family/machine/basis/preset and whose size and thread bands
        // both intersect would make a lookup ambiguous - rejected.
        for (std::size_t i = 0; i < coefficients.cells.size(); ++i)
        {
            const CoefficientCellKey& a = coefficients.cells[i].key;

            for (std::size_t j = i + 1; j < coefficients.cells.size(); ++j)
            {
                const CoefficientCellKey& b = coefficients.cells[j].key;

                if (a.pathFamily != b.pathFamily || a.machineClassKey != b.machineClassKey ||
                    a.basisFamilyId != b.basisFamilyId || a.presetId != b.presetId)
                {
                    continue;
                }

                const bool sizeOverlap =
                    a.nBasisLower <= b.nBasisUpper && b.nBasisLower <= a.nBasisUpper;
                const bool threadsOverlap =
                    a.threadsLower <= b.threadsUpper && b.threadsLower <= a.threadsUpper;

                if (sizeOverlap && threadsOverlap)
                {
                    const std::string first = CanonicalCellKey(a);
                    const std::string second = CanonicalCellKey(b);

                    if (first == second)
                    {
                        return std::unexpected(
                            Reject("coefficient file: two cells share the canonical key " + first +
                                   " (duplicate cells are rejected)"));
                    }

                    std::string message = "coefficient file: cells ";
                    message += first;
                    message += " and ";
                    message += second;
                    message += "overlap in coverage (same path_family, machine_class_key, "
                               "basis_family_id and preset_id with intersecting size and "
                               "thread bands); a lookup would be ambiguous";
                    return std::unexpected(Reject(std::move(message)));
                }
            }
        }

        // Cells load canonical-key-sorted: deterministic lookups, records
        // and diagnostics whatever order the publisher wrote.
        std::sort(coefficients.cells.begin(),
                  coefficients.cells.end(),
                  [](const CoefficientCell& a, const CoefficientCell& b) {
                      return CanonicalCellKey(a.key) < CanonicalCellKey(b.key);
                  });

        return coefficients;
    } catch (const nlohmann::json::exception& exception)
    {
        return std::unexpected(
            Reject("coefficient file: not readable JSON: " + std::string(exception.what())));
    }
}

qcx::Result<SelectionCoefficients> LoadSelectionCoefficientsFile(
    const std::filesystem::path& path) noexcept {
    try
    {
        std::ifstream stream(path);

        if (!stream)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kIOError,
                           "coefficient file \"" + path.string() + "\": cannot open for reading"});
        }

        std::ostringstream buffer;
        buffer << stream.rdbuf();

        if (stream.bad())
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kIOError,
                           "coefficient file \"" + path.string() + "\": failed while reading"});
        }

        return ParseSelectionCoefficients(buffer.str());
    } catch (const nlohmann::json::exception& exception)
    {
        return std::unexpected(
            Reject("coefficient file: not readable JSON: " + std::string(exception.what())));
    }
}

} // namespace qcx::io
