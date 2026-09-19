#pragma once

// The size-class accuracy report of the composed QFMM runs:
// the honest semantics split. The QFMM
// Coulomb half keeps its per-preset constant budgets relative to exact
// Coulomb ({1e-5, 1e-7, 1e-8} Eh, the kTight rung re-derived
// from 1e-9), but the composed run's exchange
// half is the direct builder at the preset's fixed density gate, whose
// dropped mass accumulates with the basis - so no constant total-method
// budget is claimed. Instead every composed run is reported under
// PRE-REGISTERED size-class bins (fixed before the measurements, never
// refit, identical across presets), and the per-bin aggregates print the
// fitted scaling law deltaE = C * threshold * N_bf^p with p reported.
// The report is REPORTING, never a budget; the advertised ladder is a
// prior claim, and the per-run discriminator band never chooses a preset
// a-posteriori (measured class: 2.29e-5 at C12's 86
// functions to ~1.3-1.5e-4 at C24 in the 100-199 bin).

#include "qcx/integrals/accuracy.hpp"

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace qcx::driver {

/// The pre-registered size-class bin of a basis-function count:
/// 0-99 / 100-199 / 200-399 / 400-799 / 800-1599, identical across the
/// accuracy presets and fixed A PRIORI - never refit to measured data.
/// Counts at or above 1600 have no pre-registered bin: they are outside
/// the table's claim and are reported as such.
/// \ingroup qcx-driver
enum class SizeClassBin {
    kBasis0To99 = 0, ///< 0-99 basis functions.
    kBasis100To199, ///< 100-199 basis functions.
    kBasis200To399, ///< 200-399 basis functions.
    kBasis400To799, ///< 400-799 basis functions.
    kBasis800To1599, ///< 800-1599 basis functions.
    kNoPreRegisteredBin, ///< >= 1600: outside the pre-registered table.
};

/// The largest basis count of each pre-registered bin, ascending.
/// \ingroup qcx-driver
inline constexpr std::array<int, 5> kSizeClassBinUpperBounds{99, 199, 399, 799, 1599};

/// The pre-registered size-class bin a basis-function count falls into.
/// \param basisCount The run's basis-function count N_bf (>= 0).
/// \returns The bin, or kNoPreRegisteredBin for counts the table does not
/// pre-register (negative or >= 1600).
/// \ingroup qcx-driver
SizeClassBin SizeClassBinForBasisCount(int basisCount) noexcept;

/// The display label of a size-class bin ("0-99", "100-199", ...,
/// "800-1599", "no pre-registered bin (>= 1600)").
/// \param bin The bin.
/// \returns The label.
/// \ingroup qcx-driver
std::string_view SizeClassBinLabel(SizeClassBin bin) noexcept;

/// One measured composed-QFMM run of the size-class report: the fixture,
/// its basis-function count, the accuracy preset the run ran at, the
/// near-exact reference the deviations were measured against, and the
/// absolute deviations - the composed run's total (always measured) and,
/// when the run's context measured them separately, the Coulomb half's
/// own deviation and the exchange half's own class. A row without the
/// halves' decomposition prints its bin and the composed deviation only.
///
/// The reference label names what the row is measured against (for
/// example "kTight direct" or "kNormal direct"); the composed run's
/// total deviation from a near-exact reference is the honest number for
/// the size-class table - the halves are contextual attribution.
/// \ingroup qcx-driver
struct QfmmAccuracyRunRow {
    std::string fixture; ///< The fixture label (e.g. "C12H26/STO-3G").
    int basisCount = 0; ///< The fixture's basis-function count N_bf.
    qcx::integrals::AccuracyPreset preset =
        qcx::integrals::AccuracyPreset::kNormal; ///< The run's accuracy preset.
    std::string reference; ///< The near-exact reference label (e.g. "kTight direct").
    double composedDeviation = 0.0; ///< |E_composed - E_reference|, Eh.
    std::optional<double>
        coulombHalfDeviation; ///< The Coulomb half's own |dE| (Eh), when measured.
    std::optional<double>
        exchangeHalfDeviation; ///< The exchange half's own class |dE| (Eh), when measured.
};

/// The least-squares fit of a size-class band's scaling law,
/// deltaE = C * threshold * N_bf^p (log-space fit over the bin's
/// composed-deviation rows). Reported only when the bin holds at least
/// three fixtures (the n < 3 underpopulation rule - no fit is
/// claimed from fewer points).
/// \ingroup qcx-driver
struct SizeClassScalingFit {
    bool valid = false; ///< False when fewer than three rows or any row lacks a positive deviation.
    double prefactor = 0.0; ///< The fitted C.
    double exponent = 0.0; ///< The fitted p (the reported exponent).
    double threshold = 0.0; ///< The preset's density gate the fit is anchored to.
};

/// The per-run size-class band line of one measured composed run: the
/// fixture's a-priori bin, the basis count, the preset and reference,
/// and the measured deviations (composed always; the Coulomb and exchange
/// halves when the run's context measured them).
/// \param row The measured run.
/// \returns The band line (text; the report never enters the result JSON).
/// \ingroup qcx-driver
std::string FormatSizeClassBandLine(const QfmmAccuracyRunRow& row);

/// The aggregate size-class band table of a set of measured composed
/// runs, formatted per the pre-registered-bin schema: for each
/// (preset, reference) group and each occupied bin - n_fixtures, the
/// N_basis range, the max/median/95th-percentile absolute composed
/// deviation (Eh), the worst fixture, and the fitted scaling law
/// deltaE = C * threshold * N_bf^p (C and p). Bins holding fewer than
/// three fixtures are marked UNDERPOPULATED and carry no fit. Empty
/// bins are omitted; an empty row set returns the empty string.
/// \param rows The measured runs (any order; grouping and bins are derived).
/// \returns The table text, or "" when \p rows is empty.
/// \ingroup qcx-driver
std::string FormatSizeClassBandTable(const std::vector<QfmmAccuracyRunRow>& rows);

} // namespace qcx::driver
