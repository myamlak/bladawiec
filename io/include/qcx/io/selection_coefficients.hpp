// The engine-side selection-coefficient file (vocabulary of the fit
// corpus, tools/bench/cost_table_schema.py): the
// published measured-cost table the driver's selection heuristic
// (qcx/driver/selection_heuristic.hpp) consults STRICTLY AFTER memory
// admission, for admitting candidates only. The loader lives in io (the
// schema layer; nlohmann is an io PRIVATE dependency); the seam policy
// lives in the driver. Loaded once per run at driver start - the load-time
// validation is the increment's load-time cost, negligible against the HF
// run it precedes.
//
// File contract (schema_version 1, exact match only). A JSON object with
// exactly these keys: schema_version (int, must equal
// kSelectionCoefficientsSchemaVersion), file_version (the publisher's
// version id of this file - what the run record reports), fit_metadata,
// cells. Any schema-version mismatch, unknown key, malformed value, or
// broken invariant below rejects the WHOLE file (fail-closed): a partially
// applied table must never half-drive a selection. A rejected file behaves
// exactly like no file - the analytic heuristic stands
// byte-identical.
//
// fit_metadata: condition_number (finite, 1..kCoefficientMaxConditionNumber
// - an ill-conditioned fit is rejected whole), active_rank (>= 1),
// aliased_directions (named aliased directions of the fit), covariance
// (the joint covariance, validated finite/square/symmetric/PSD - validated
// because the fit published it, never used for selection),
// interpolation_enabled (the same-family nearest size-bin interpolation of
// the fallback chain is NOT implemented in the engine seam; a file that
// enables it is rejected whole rather than silently under-applied).
//
// cells: the published cells, each with a key (path_family from the closed
// kCoefficientPathFamilies ladder set; machine_class_key, basis_family_id
// and preset_id as corpus tokens; the inclusive n_basis and threads bands)
// and EXACTLY ONE payload: the measured payload (median, mad,
// n_measurements) or the borrowed payload (borrowed.donor_family - never
// the cell's own family - and borrowed.inflation > 0). Both payloads carry
// staleness_version (>= 0) and drift_state ("normal" | "stale" | "frozen").
// Duplicate canonical keys and coverage-overlapping cells are rejected
// (a lookup must be unambiguous); cells load canonical-key-sorted so every
// lookup and record is deterministic. Unknown cells (other machines,
// families, basis families, presets, size or thread bands) are tolerated:
// they simply never match this run's lookups.
//
// Drift verdicts are the fit's - the engine never re-validates. "stale"
// skips the cell individually (the table stays usable); "frozen"
// (drift-frozen) is permanently ineligible until a re-fitted file
// publishes the cell anew - never re-validated without a re-fit.

#pragma once

#include "qcx/error.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace qcx::io {

/// The coefficient file's schema version. Exact match only: any other
/// value rejects the file whole (fail-closed) - a version-mismatched table
/// must never half-apply. Bumped when the FILE contract changes; the run
/// result schema's RunResult::kSchemaVersion is a different pin (the
/// record side).
/// \ingroup qcx-io
inline constexpr int kSelectionCoefficientsSchemaVersion = 1;

/// The closed ladder-path family set (the fit corpus's PATH_FAMILIES,
/// tools/bench/cost_table_schema.py): direct_screened, light_path, ri_j,
/// qfmm, gpu. Every path_family token in the file is validated against
/// this set at load.
/// \ingroup qcx-io
inline constexpr std::array<std::string_view, 5> kCoefficientPathFamilies = {
    "direct_screened", "light_path", "ri_j", "qfmm", "gpu"};

/// The fit's hard condition-number ceiling ("~1e6"): a
/// fit_metadata.condition_number above it means the fit is ill-conditioned
/// and the file is rejected whole.
/// \ingroup qcx-io
inline constexpr double kCoefficientMaxConditionNumber = 1e6;

/// A cell's drift verdict (published by the fit's drift loop; the engine
/// never re-validates a verdict).
/// \ingroup qcx-io
enum class CoefficientDrift {
    kNormal, ///< The cell is current and usable.
    kStale, ///< The fit flagged drift: the cell is individually ineligible
            ///< (the table stays usable); a lookup skips it.
    kFrozen, ///< Drift-frozen: permanently ineligible until a re-fitted
             ///< file publishes the cell anew (never re-validated without a
             ///< re-fit). Not stale.
};

/// The lookup key of a coefficient cell. The canonical key string is
/// "{family}@{machine}@{basis}[{nBasisLower}:{nBasisUpper}]@{preset}@t[{threadsLower}:{threadsUpper}]",
/// the engine-side mirror of the fit corpus's canonical
/// "path@machine@basis[lower:upper]@composition" key with the run's
/// preset and thread band appended (no composition model exists on the
/// engine side).
/// \ingroup qcx-io
struct CoefficientCellKey {
    std::string pathFamily; ///< One of kCoefficientPathFamilies.
    std::string machineClassKey; ///< The calibration node's machine-class key
                                 ///< (the fit corpus vocabulary).
    std::string basisFamilyId; ///< The basis-family id the cell calibrates
                               ///< (the fit corpus vocabulary).
    int nBasisLower = 1; ///< Inclusive size-bin lower edge (orbital basis
                         ///< functions).
    int nBasisUpper = 1; ///< Inclusive size-bin upper edge.
    std::string presetId; ///< The method preset the cell calibrates;
                          ///< "default" = the base preset (v1's only preset).
    int threadsLower = 1; ///< Inclusive thread-band lower edge.
    int threadsUpper = 1; ///< Inclusive thread-band upper edge.
};

/// The borrowed-cell payload (the second fallback): the new-family
/// placeholder, eligible WITH the stored inflation when its donor cell
/// resolves.
/// \ingroup qcx-io
struct CoefficientBorrow {
    std::string donorFamily; ///< The measured family the cell borrows from:
                             ///< a member of kCoefficientPathFamilies, never
                             ///< the cell's own family.
    double inflation = 1.0; ///< > 0; the multiplier applied to the donor
                            ///< cell's score.
};

/// One published cell: a measured cell (median/mad/nMeasurements) or a
/// borrowed cell (borrow) - exactly one of the two payloads. Every cell
/// carries the fit's staleness counter and drift verdict.
/// \ingroup qcx-io
struct CoefficientCell {
    CoefficientCellKey key; ///< The cell's lookup key.
    /// The measured center (the score = median + 2 mad is the seam's);
    /// measured cells only.
    std::optional<double> median;
    /// The robust spread (mad, >= 0); measured cells only.
    std::optional<double> mad;
    int nMeasurements = 0; ///< The measurements behind the published stats;
                           ///< measured cells only, >= 1.
    int stalenessVersion = 0; ///< The fit's staleness counter, >= 0 (the
                              ///< engine never re-validates; the drift loop
                              ///< reads it).
    CoefficientDrift drift = CoefficientDrift::kNormal; ///< The drift verdict.
    std::optional<CoefficientBorrow> borrow; ///< The borrowed payload;
                                             ///< borrowed cells only.
};

/// The file's fit-metadata block. Validated strictly at load and carried
/// for the audit trail; never used for selection (selection consumes the
/// cells).
/// \ingroup qcx-io
struct CoefficientFitMetadata {
    double conditionNumber = 1.0; ///< Validated 1..kCoefficientMaxConditionNumber.
    int activeRank = 1; ///< The fit's active rank, >= 1.
    /// The named aliased directions of the fit (degenerate parameter
    /// directions the publisher resolved or excluded).
    std::vector<std::string> aliasedDirections;
    /// The joint covariance of the fit, validated finite/square/symmetric/
    /// PSD at load. Not used for selection.
    std::vector<std::vector<double>> covariance;
    bool interpolationEnabled = false; ///< Always false in schema 1: a file
                                       ///< enabling interpolation is rejected
                                       ///< (the engine has no interpolation).
};

/// A loaded coefficient file: the validated model behind the JSON
/// contract. Immutable after load; the driver's seam reads it through
/// FindCoefficientCell.
/// \ingroup qcx-io
struct SelectionCoefficients {
    int schemaVersion = kSelectionCoefficientsSchemaVersion; ///< The validated
                                                             ///< schema version.
    std::string fileVersion; ///< The publisher's version id of this file
                             ///< (what the run record reports).
    CoefficientFitMetadata fitMetadata; ///< The validated fit-metadata block.
    /// Every cell of the file, canonical-key-sorted at load; pairwise
    /// disjoint in coverage (a lookup is unambiguous: at most one cell
    /// matches).
    std::vector<CoefficientCell> cells;
};

/// The canonical cell-key string (see CoefficientCellKey): a deterministic
/// token join used by the run record and the diagnostic notes.
/// \param key The cell key.
/// \returns The canonical string.
/// \ingroup qcx-io
std::string CanonicalCellKey(const CoefficientCellKey& key);

/// The partial-key cell lookup the seam uses: the cell whose path family,
/// machine-class key, basis-family id and preset id equal the query and
/// whose size bin and thread band cover the run's nBasis/threads. The
/// load-time disjoint-coverage validation makes the match unique when it
/// exists.
/// \param coefficients The loaded file.
/// \param pathFamily The candidate's ladder family (one of
/// kCoefficientPathFamilies).
/// \param machineClassKey The run's machine-class key.
/// \param basisFamilyId The run's basis-family id.
/// \param presetId The run's preset id.
/// \param nBasis The run's orbital basis-function count.
/// \param threads The run's effective thread count.
/// \returns The matching cell, or nullptr when the file has no cell for
/// this query.
/// \ingroup qcx-io
const CoefficientCell* FindCoefficientCell(const SelectionCoefficients& coefficients,
                                           std::string_view pathFamily,
                                           std::string_view machineClassKey,
                                           std::string_view basisFamilyId,
                                           std::string_view presetId,
                                           std::size_t nBasis,
                                           int threads) noexcept;

/// Parses and load-validates a coefficient file's JSON text (the file
/// contract in the header note). Pure: no I/O. Every contract violation
/// rejects the whole file with a kInvalidArgument error naming the broken
/// element; the file's cells are never partially applied.
/// \param json The file's JSON text.
/// \returns The validated model, or the rejection.
/// \ingroup qcx-io
qcx::Result<SelectionCoefficients> ParseSelectionCoefficients(std::string_view json) noexcept;

/// Loads, parses and load-validates a coefficient file. The one-time
/// per-run load of the seam; file-reading failures (missing/unreadable
/// file) return kIOError, contract violations kInvalidArgument (the same
/// rejections ParseSelectionCoefficients reports).
/// \param path The file path.
/// \returns The validated model, or the failure.
/// \ingroup qcx-io
qcx::Result<SelectionCoefficients> LoadSelectionCoefficientsFile(
    const std::filesystem::path& path) noexcept;

} // namespace qcx::io
