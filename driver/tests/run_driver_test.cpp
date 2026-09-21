// The driver acceptance test: RunDriver reproduces the
// H2/H2O RHF pins and the O2-triplet UHF pin from real input-file text -
// energies, spin-squared, and the populations/moments block - and rejects
// the schema's not-yet-wired combinations. The pins mirror direct_rhf_test
// .cpp / direct_uhf_test.cpp / populations_test.cpp / multipoles_test.cpp
// (same geometries, kTight, defaults or the explicit use_diis = false the
// O2 landscape needs), but the input reaches the SCF through the full
// driver path - TOML -> RunInput -> molecule/basis load -> one-electron
// integrals -> seam -> properties -> JSON.

#include "alkane_sto3g.hpp"
#include "cap_child_harness.hpp"
#include "fast_test_mode.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/backend/gpu_compute_profile.hpp"
#include "qcx/backend/host_compute_profile.hpp"
#include "qcx/driver/accuracy_class_report.hpp"
#include "qcx/driver/basis_counts.hpp"
#include "qcx/driver/run_driver.hpp"
#include "qcx/driver/selection_heuristic.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/precision_policy.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/io/result_json.hpp"
#include "qcx/io/run_input.hpp"
#include "qcx/memory/allocation_instrument.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#if defined(QcxHasStorage)
#include "qcx/storage/scf_checkpoint.hpp"
#endif

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <numbers>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using qcx::driver::RunDriver;
using qcx::io::ParseRunInput;

// The H2 geometry of h2_sto3g.hpp in Angstrom: R = 1.4 Bohr on the x axis
// (0.74084809526419992 A after the exact conversion) - the pin's geometry.
const char* kH2Toml = R"(
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
// Measured 2026-09-16: this gate is VESTIGIAL - the operating default (1e-8/1e-6,
// the ladder's kNormal rung) was run against the committed 1e-10/1e-10 pair on the
// driver CLI and the two records differ in nothing the cell asserts on: energy
// -1.116714325175768 (2.42e-11 from the pinned -1.1167143252, band 1e-8), 2
// iterations, converged true, and every resources_resolved/builder
// field byte-identical. H2/STO-3G converges in two passes, so the walk lands on the
// same fixed point from either gate and the tight pair was copy-paste, not a bound.

// kH2Toml with the [resources] thread ceiling pinned to 1: the single-
// chunk serial path. The [resources] block must be a top-level table, so
// this is its own text rather than an append.
const char* kH2TomlThreadCap1 = R"(
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

[resources]
thread_cap = 1
)";
// Measured 2026-09-16: VESTIGIAL, and the cell's subject - the serial path's
// bit-identity pin - is gate-INDEPENDENT (both legs share this gate, so it
// cancels). Relaxed with its sibling above: the run at the default reproduces
// -1.116714325175768 and converged true, the same record the 1e-10 pair gave.

// kH2Toml with the [method] fock_builder line REMOVED: an absent builder
// key at nBasis 2 <= the 1000-basis-function lean boundary is the ladder's
// bottom tier (the lean Schwarz-only member of the direct family, the
// 2026-09-13 ruling - the run record carries the tier table).
// The explicit "direct" of kH2Toml stays the machinery at every size
// (Ruling B), so the two fixtures are the flag's true/false pair. The SCF
// thresholds are the OPERATING defaults (1e-8/1e-6): the 2026-09-04 rule
// retires 1e-10 thresholds for new runs, and this fixture asserts the
// WIRING - the physics of the lean path is pinned by the sibling
// H2DirectRhfReproducesThePin.
const char* kH2TomlAutoLean = R"(
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

// kH2TomlAutoLean with the explicit `fock_builder = "lean"` word: the
// SECOND spelling of the same within-family member - the opt-in that
// reaches the lean builder at ANY size, the >1000-basis-function admission
// (above the ceiling the absent key resolves to the machinery). At nBasis 2
// both spellings wire the same builder, so this fixture's record must agree
// with kH2TomlAutoLean's - that agreement is the wiring pin on the small
// side, and the above-ceiling discrimination rides the large-molecule fixture of
// LeanWordAdmitsTheLeanMemberAboveTheCeiling (1210 basis functions; a
// converged above-ceiling lean run stays ladder-harness coverage).
const char* kH2TomlLeanWord = R"(
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

// The water geometry of h2o_sto3g.hpp in Angstrom: O at the origin, H at
// (+-d, h, 0), d = 1.430428808 / kAngstromToBohr, h = 1.107157044 /
// kAngstromToBohr (r(OH) = 0.9572 A, angle = 104.52 deg) - the round trip
// reproduces the fixture's Bohr values to double precision.
const char* kH2oToml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kTight"
)";

// kH2oToml with the routing threshold switched from kTight's 0.0 to
// kNormal's 1e-10: the same water/STO-3G machinery run with the fp32
// lane ADMITTED rather than switched off. It still routes nothing - the
// a-priori gate needs a pair far apart enough that C_class * 1e-7 * Q_bra
// * Q_ket * dMax <= 1e-10, and no water pair is - so this is the fixture
// that shows the true zero arising from the gate's arithmetic rather than
// from the preset being off (see CertifiedBoundAlkaneToml for the alkane
// that does route).
const char* kH2oNormalToml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[scf]
energy_tolerance = 1e-8
density_tolerance = 1e-6
use_diis = true
)";

// The O2 triplet of o2_sto3g.hpp in Angstrom: R = 2.2818443 Bohr on the z
// axis (1.2075000023889084 A after the exact conversion). The direct UHF
// path needs the SAD seed; per-spin DIIS is fine on top of it and lands on
// the pinned ground state (the properties module's own O2 recipe,
// o2_sad_guess.hpp, runs the same combination).
const char* kO2Toml = R"(
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
fock_builder = "direct"
accuracy = "kTight"

[scf]
use_diis = true
energy_tolerance = 1e-8
density_tolerance = 1e-6

[guess]
type = "sad"
)";
// Measured 2026-09-16 and VESTIGIAL: the operating default was run against the
// committed 1e-10/1e-10 pair on the driver CLI and every assertion below holds at
// both. Tight -147.63394682039947 / spin-squared 2.003410846638319 / 45 iterations;
// default -147.63394682039942 / 2.0034107634059666 / 34. Against the pins
// (-147.63394678545018 at 5e-7, 2.0034108576810308 at 1e-6) the default sits 3.49e-8
// and 9.43e-8 away - 14x and 11x inside. The tight pair was a the defaults rule note about when
// the pin was RECORDED, not a bound the run needs: the same fixed point is reached.

// The composed-QFMM mirror of kO2Toml: the same pinned recipe with
// fock_builder = "qfmm" (the QFMM schema knobs stay absent - the preset
// defaults of the engine contract; the per-spin K half is the
// exchange-only direct builder inside the composed QfmmHfFockBuilder).
// The composition must reproduce the direct pin: at kTight the QFMM error
// is far inside the 5e-7 band that already covers the DIIS/thread fixed-
// point cluster of the direct run, and the SAD-seeded plain iteration
// lands on the same pinned ground state (the direct-builder UHF assembly landscape rule - the
// composed UHF runs plain like the direct path).
const char* kO2QfmmToml = R"(
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
fock_builder = "qfmm"
accuracy = "kTight"

[scf]
use_diis = true
energy_tolerance = 1e-8
density_tolerance = 1e-6

[guess]
type = "sad"
)";
// Measured 2026-09-16 and VESTIGIAL: at the operating default the composed run
// gives -147.63394682039942 (34 iterations) against the 5e-7 band's
// -147.63394678545018, a 3.49e-8 distance - identical to the tight gate's
// -147.6339468203995 at 63 iterations. The QFMM half's own error is what this
// cell bounds, and it is far inside the band from either gate.

// The unrestricted ri_j_link mirror of kO2Toml (the per-spin adapter): the
// same pinned recipe with fock_builder = "ri_j_link", the family whose
// per-spin adapter landed here. The SAD seed and the per-spin DIIS are
// the direct pin's verbatim - what differs is the Fock assembly, so the run
// proves the adapter reaches the SAME UHF branch on an OPEN-shell deck
// (alpha != beta): the spin structure below is read from a genuinely
// polarized density pair, not from the symmetric case.
//
// The [scf] gates are the OPERATING DEFAULTS (1e-8/1e-6), deliberately not
// kO2Toml's explicit 1e-10 pair: that pair is not a rung of the ruled gate
// ladder, and a line this commit ADDS at it is exactly what the
// gate-tolerance hook check refuses (tools/check_gate_tolerances.py - the
// pre-existing lines in kO2Toml are printed, not failed). The default-gate
// run was MEASURED to reach the same branch: E = -147.6352140935264 (34
// iterations) against the tight gate's -147.63521409352643 (63), spin-squared
// 2.0034095906770197 against 2.0034096861828736 (9.5e-8 apart), per-atom
// spin 1.0 both ways.
const char* kO2RiJLinkToml = R"(
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
fock_builder = "ri_j_link"
accuracy = "kTight"

[scf]
use_diis = true

[guess]
type = "sad"
)";

// The unrestricted composed-full-RI mirror of kO2Toml (the per-spin adapter
// the pair entry point's other half landed): the same pinned recipe with
// fock_builder = "ri_jk", the family whose per-spin adapter landed here.
// The SAD seed, the tight gate and the per-spin DIIS are the direct pin's
// verbatim, so what differs is the Fock assembly alone and the run proves the
// adapter reaches the SAME UHF branch on an OPEN-shell deck (alpha != beta) -
// and the aux it resolves is this KIND's (def2-universal-jkfit), not the
// ri_j_link family's J-fit.
//
// The iteration cap is the ONE departure from the sibling fixture, and it is
// MEASURED rather than precautionary: at the [scf] defaults (100, the io
// member's own default) this deck does NOT converge - the run stops at the cap
// with energy_delta 5.7e-4 and rms_density_delta 6.7e-4, four orders outside
// the gate it is walking toward - while the ri_j_link sibling, which fits the
// spin-independent Coulomb half instead, converges in 63. The fitted EXCHANGE
// is what makes the near-degenerate O2 landscape slower to walk, so the cap is
// raised rather than the gate loosened (the defaults rule's thresholds are not to be tuned
// for a fixture) and the converging iteration count is printed by the pin
// below.
const char* kO2RiJkToml = R"(
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
fock_builder = "ri_jk"
accuracy = "kTight"

[scf]
use_diis = true
max_iterations = 400

[guess]
type = "sad"
)";
// The lean auto-lean mirror of kO2Toml (the unrestricted seam): the identical
// pinned recipe with the `fock_builder` key ABSENT, so the auto rule reads
// it - at nBasis 10 <= the 1000-basis-function ceiling the UHF path wires
// the direct family's lean member, exactly as the RHF path already does.
// The key's absence is the fixture's whole content; everything else - the
// SAD seed, the tight gate, the per-spin DIIS - is the direct pin's.
const char* kO2AutoLeanToml = R"(
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
energy_tolerance = 1e-8
density_tolerance = 1e-6

[guess]
type = "sad"
)";
// Measured 2026-09-16 and VESTIGIAL: at the operating default the lean member
// gives -147.6339468204004 (34 iterations) and spin-squared 2.0034107635512246.
// The cell's energy assertion is an ORDERING - the engaged lean must land closer
// to the machinery pin than the disengaged lean's 2.2120e-7 - and the default's
// 3.4950e-8 clears it 6.3x; its spin-squared band (2.003410846644341 at 1e-6) is
// cleared 12x. The tight gate's own numbers (63 iterations, spin-squared
// 2.0034108574627005) sit no closer to the ordering bar.

// The explicit `fock_builder = "lean"` spelling on the same UHF recipe:
// one within-family request, no family word (the parse site keeps "lean"
// out of the builder slot), admitted at any size - the RHF spelling's
// UHF twin. The two fixtures must reach the same member and the same
// numbers (the seam has one arm, not two spellings of it).
const char* kO2LeanWordToml = R"(
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
fock_builder = "lean"
accuracy = "kTight"

[scf]
use_diis = true
energy_tolerance = 1e-8
density_tolerance = 1e-6

[guess]
type = "sad"
)";
// Measured 2026-09-16 and VESTIGIAL: the cell's assertions are the record facts
// (lean true, builder direct, the honored-request text) plus an energy EXACTLY
// equal to the auto-lean sibling's. At the operating default both fixtures give
// -147.6339468204004 in 34 iterations, so the equality the cell exists for still
// holds; the 1e-10 pair bought nothing it checks.

// The H2O [properties] block with every analyzer on:
// charges (Hirshfeld/Voronoi), ESP (CHELPG), EDDB, and the
// Nalewajski-Mrozek bond orders on one run.
const char* kH2oStage72Toml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kTight"

[properties]
hirshfeld = true
voronoi = true
esp = "chelpg"
eddb = true
nalewajski = true
)";

// The H2O NOCV partition written in FILE order - the O atom alone plus the
// two H atoms. The driver maps the groups onto the canonical [H, H, O]
// order (Molecule canonicalization); the mapping is the point of this
// input.
const char* kH2oNocvToml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kTight"

[scf]
# The pins were recorded at the tight gate; 2026-09-16 MEASURED that the gate is not what holds them - at the operating default every cell assertion still holds, worst margin 3.2x over 3 identical runs, and the ~5e-6 figure once cited for this fixture belongs to properties/tests/nocv_test.cpp. Relaxed to the operating default (owner ruling); the pins are unchanged and still explicit.
energy_tolerance = 1e-08
density_tolerance = 1e-06

[properties]
nocv_fragments = [[0], [1, 2]]
)";

// The H2 exactly-paired partition: each H its own fragment (canonical
// order equals file order here - both rows already sorted).
const char* kH2NocvToml = R"(
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

[properties]
nocv_fragments = [[0], [1]]
)";

// The H2O Fukui response through the driver path: the neutral RHF singlet
// plus the two UHF doublet charged species the driver runs itself.
const char* kH2oFukuiToml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kTight"

[properties]
fukui = true
)";

// Extracts the number after the "key" member of the dumped JSON document
// (nlohmann dump(2) emits `"key": value`), scanning from \p from (0 = the
// whole document) so a caller can scope a read into a nested block whose
// member name an earlier sibling block also carries. Returns nullopt when
// the key is absent at or after \p from.
std::optional<double> JsonNumber(const std::string& json,
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

    // The null policy: an unset optional member is emitted as null; treat
    // it as absent instead of letting std::stod throw on "null".
    std::size_t afterColon = valuePos + 1;

    while (afterColon < json.size() && json[afterColon] == ' ')
    {
        ++afterColon;
    }

    if (afterColon < json.size() && json[afterColon] == 'n')
    {
        return std::nullopt;
    }

    return std::stod(json.substr(valuePos + 1));
}

// The doubles of the first one-level JSON array following \p pattern (e.g.
// "\"total\":" — the trailing colon disambiguates "total_energy_hartree")
// at or after \p from, plus the position after the array so callers can
// chain searches into a nested block (e.g. lowdin's "total" after
// "\"lowdin\""). Returns nullopt when the pattern or array is absent.
// One-level arrays only: a nested matrix's row-opening '[' would land
// inside the first element's substring and std::stod would reject it -
// use JsonMatrixRow0 for those.
struct JsonArrayMatch {
    std::vector<double> values; ///< The array elements.
    std::size_t nextPosition = 0; ///< One past the closing ']'.
};

std::optional<JsonArrayMatch> JsonNumberArray(const std::string& json,
                                              std::string_view pattern,
                                              std::size_t from = 0) {
    const auto patternPos = json.find(pattern, from);

    if (patternPos == std::string::npos)
    {
        return std::nullopt;
    }

    // The null policy: an unset optional member is emitted as null; treat
    // it as absent instead of scanning on to the next array in the
    // document (which would silently serve a different block's values).
    std::size_t valuePos = patternPos + pattern.size();

    while (valuePos < json.size() && json[valuePos] == ' ')
    {
        ++valuePos;
    }

    if (valuePos < json.size() && json[valuePos] == 'n')
    {
        return std::nullopt;
    }

    const auto open = json.find('[', patternPos);

    if (open == std::string::npos)
    {
        return std::nullopt;
    }

    const auto close = json.find(']', open);

    if (close == std::string::npos || open + 1 >= close)
    {
        // The latter: an empty array - std::stod("") would throw, and no
        // caller probes an array that may legitimately be empty.
        return std::nullopt;
    }

    std::vector<double> values;
    std::size_t start = open + 1;

    for (;;)
    {
        const auto comma = json.find(',', start);
        const auto end = (comma == std::string::npos || comma > close) ? close : comma;
        values.push_back(std::stod(json.substr(start, end - start)));

        if (comma == std::string::npos || comma > close)
        {
            break;
        }

        start = comma + 1;
    }

    return JsonArrayMatch{std::move(values), close + 1};
}

// A scalar number emitted after \p pattern at or after \p from: the driver
// emits the bond-critical-point members density, laplacian and ellipticity
// as bare numbers, and JsonNumberArray's forward scan would serve the next
// array in the document (eigenvalues or position_bohr - the std::map-
// backed serializer sorts the keys) instead of the scalar itself. Returns
// nullopt when the pattern is absent or the member is emitted null (the
// null policy: unset optional members read as absent, never as zero).
std::optional<double> JsonScalarNumber(const std::string& json,
                                       std::string_view pattern,
                                       std::size_t from = 0) {
    const auto patternPos = json.find(pattern, from);

    if (patternPos == std::string::npos)
    {
        return std::nullopt;
    }

    std::size_t valuePos = patternPos + pattern.size();

    while (valuePos < json.size() && json[valuePos] == ' ')
    {
        ++valuePos;
    }

    if (valuePos < json.size() && json[valuePos] == 'n')
    {
        return std::nullopt;
    }

    const auto end = json.find_first_of(",]}", valuePos);

    if (end == std::string::npos || valuePos >= end)
    {
        return std::nullopt;
    }

    return std::stod(json.substr(valuePos, end - valuePos));
}

// Row 0 of the first nested JSON array following \p pattern at or after
// \p from: the flattened JsonNumberArray probe cannot parse a matrix (the
// row-opening '[' would land inside the first element's substring), and
// the driver emits the nalewajski/mayer bond-order matrices as
// `[[...], [...], ...]`. Row 0 is what the module-level pins assert for
// the [H, H, O] atom order (the O-H and H-H entries of the reference
// atom). Returns nullopt when the pattern or the matrix is absent.
std::optional<std::vector<double>> JsonMatrixRow0(const std::string& json,
                                                  std::string_view pattern,
                                                  std::size_t from = 0) {
    const auto patternPos = json.find(pattern, from);

    if (patternPos == std::string::npos)
    {
        return std::nullopt;
    }

    // The null policy, as JsonNumberArray: an unset member is null, not an
    // absent probe target; scanning past it would serve the next matrix.
    std::size_t valuePos = patternPos + pattern.size();

    while (valuePos < json.size() && json[valuePos] == ' ')
    {
        ++valuePos;
    }

    if (valuePos < json.size() && json[valuePos] == 'n')
    {
        return std::nullopt;
    }

    const auto outerOpen = json.find('[', patternPos);

    if (outerOpen == std::string::npos)
    {
        return std::nullopt;
    }

    const auto rowOpen = json.find('[', outerOpen + 1);

    if (rowOpen == std::string::npos)
    {
        return std::nullopt;
    }

    const auto rowClose = json.find(']', rowOpen);

    if (rowClose == std::string::npos || rowOpen + 1 >= rowClose)
    {
        // The latter: an empty matrix row - std::stod("") would throw.
        return std::nullopt;
    }

    std::vector<double> values;
    std::size_t start = rowOpen + 1;

    for (;;)
    {
        const auto comma = json.find(',', start);
        const auto end = (comma == std::string::npos || comma > rowClose) ? rowClose : comma;
        values.push_back(std::stod(json.substr(start, end - start)));

        if (comma == std::string::npos || comma > rowClose)
        {
            break;
        }

        start = comma + 1;
    }

    return values;
}

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

// The composed builder composed-QFMM driver rows' [method] retune: a copy of a base
// TOML with the builder and accuracy tokens swapped (the bases pin
// fock_builder = "direct" and accuracy = "kTight"). The base constants'
// texts are guarantees of this file, and every row of a comparison is
// built from the SAME retuned text, so the parse quantization is
// identical across the row's runs.
std::string WithBuilderAndAccuracy(const char* baseToml,
                                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                   std::string_view builder,
                                   std::string_view accuracy) {
    constexpr std::string_view kBuilderToken = "fock_builder = \"direct\"";
    constexpr std::string_view kAccuracyToken = "\"kTight\"";
    std::string text(baseToml);

    const auto builderPos = text.find(kBuilderToken);
    const auto accuracyPos = text.find(kAccuracyToken);

    if (builderPos == std::string::npos || accuracyPos == std::string::npos)
    {
        ADD_FAILURE() << "base TOML lacks the [method] tokens: " << baseToml;
        return text;
    }

    // Replace the LATER position first: both positions were computed on the
    // original text, and the earlier-position replacement shifts everything
    // after it - a second replace at the stale later position corrupts the
    // text (the pre-fix rows' "expected a comment or whitespace, saw 'k'").
    const std::string builderReplacement = "fock_builder = \"" + std::string(builder) + "\"";
    const std::string accuracyReplacement = "\"" + std::string(accuracy) + "\"";

    if (builderPos < accuracyPos)
    {
        text.replace(accuracyPos, kAccuracyToken.size(), accuracyReplacement);
        text.replace(builderPos, kBuilderToken.size(), builderReplacement);
    } else
    {
        text.replace(builderPos, kBuilderToken.size(), builderReplacement);
        text.replace(accuracyPos, kAccuracyToken.size(), accuracyReplacement);
    }

    return text;
}

// The [molecule] block of an alkane-chain TOML: the shared fixture
// geometry (MakeAlkaneSto3g - the accuracy sweep's fixture, so every text
// built here sits on the recorded molecules), the coordinates in Angstrom
// (the driver TOML convention) at full round-trip precision
// (setprecision(17) - the parse quantization is identical for every text
// built from one molecule).
void WriteAlkaneMoleculeBlock(std::ostringstream& text, const qcx::molecule::Molecule& molecule) {
    const auto& coordinates = molecule.CoordinatesBohr();
    text << "[molecule]\ncharge = 0\nmultiplicity = 1\natoms = [\n";

    for (std::size_t atom = 0; atom < molecule.Atoms().size(); ++atom)
    {
        text << "    [\"" << molecule.Atoms()[atom].symbol << "\", " << std::setprecision(17)
             << coordinates(atom, 0) / qcx::molecule::kAngstromToBohr << ", "
             << coordinates(atom, 1) / qcx::molecule::kAngstromToBohr << ", "
             << coordinates(atom, 2) / qcx::molecule::kAngstromToBohr << "],\n";
    }

    text << "]\n";
}

// The certified-bound fixture: an n-carbon STO-3G RHF text on the shared
// alkane geometry, wiring the direct machinery at kNormal (the live
// routing threshold 1e-10) with the OPERATING SCF thresholds (1e-8/1e-6,
// the 2026-09-04 rule - not the sibling chain helper's 1e-10 gate). The
// fixture must be an alkane rather than this file's water/H2 pins: the
// fp32 routing gate is a-priori (C_class * 1e-7 * Q_bra * Q_ket * dMax <=
// 1e-10), and on a molecule as small as water no pair clears it, so the
// bound would read its true zero and could not distinguish a live lane
// from a dead one. Measured on this fixture family (the lane's own probe,
// 2026-09-11): methane (9 basis functions) routes zero quartets at
// kNormal, ethane (16) routes 487, butane (30) 17056 - the lane engages
// with molecular size, which is why the non-zero pin sits on butane.
std::string CertifiedBoundAlkaneToml(const qcx::molecule::Molecule& molecule,
                                     std::string_view methodExtraKeys = {}) {
    std::ostringstream text;
    WriteAlkaneMoleculeBlock(text, molecule);
    text << "\n[basis]\norbital = \"sto-3g\"\n\n[method]\ntype = \"rhf\"\nfock_builder = "
            "\"direct\"\naccuracy = \"kNormal\"\n"
         << methodExtraKeys
         << "\n[scf]\n"
            "energy_tolerance = 1e-8\ndensity_tolerance = 1e-6\nuse_diis = true\n";
    return text.str();
}

// The certified-bound enforcement fixture: the butane/STO-3G text above
// with the [method] enforce_certified_bound key appended. Same molecule,
// basis, preset and SCF thresholds as CertifiedBoundAlkaneToml, so the two
// runs differ in exactly one key - which is what makes the enforcement's
// effect (and the absence of any effect on the other keys) attributable.
std::string CertifiedBoundEnforcedAlkaneToml(const qcx::molecule::Molecule& molecule) {
    return CertifiedBoundAlkaneToml(molecule, "enforce_certified_bound = true\n");
}

// The enforcement fixture with the lane FORCED on: the same
// butane/STO-3G text as CertifiedBoundEnforcedAlkaneToml plus
// [method] force_certified_lane. It exists because the enforcement's ON arm -
// the comparison that refuses the routing and falls back to fp64 - needs a
// live fp32 lane, and on a machine whose probe resolves OFF the enforcement
// key alone runs vacuously (the disclosure rule-legal, and the reading
// CertifiedBoundEnforcementDecidesTheRouting pins). The forcing key is what
// makes that arm reachable from an input file rather than from a source
// injection of the probe's return value.
std::string CertifiedBoundEnforcementForcedAlkaneToml(const qcx::molecule::Molecule& molecule) {
    return CertifiedBoundAlkaneToml(
        molecule, "force_certified_lane = true\nenforce_certified_bound = true\n");
}

// The certified-bound ri_j fixture: the H2O/def2-SVP text of the ri_j_link
// pins (24 basis functions, the auto-selected universal-J aux - no
// [basis].aux key) with the builder word substituted, so one molecule,
// basis and preset carry both the link and the machinery leg. The pair is
// what makes the absence assertion attributable to the route rather than
// to a broken run.
std::string H2oDef2SvpToml(std::string_view builder) {
    std::string text = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "def2-svp"

[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kLoose"
)";
    constexpr std::string_view kBuilderToken = "fock_builder = \"ri_j_link\"";
    const auto builderPos = text.find(kBuilderToken);

    if (builderPos == std::string::npos)
    {
        ADD_FAILURE() << "the fixture text lacks the [method] builder token";
        return text;
    }

    text.replace(
        builderPos, kBuilderToken.size(), "fock_builder = \"" + std::string(builder) + "\"");
    return text;
}

// H2oDef2SvpToml with the method word swapped to the unrestricted spelling
// (multiplicity 1, so alpha == beta): the deck whose two legs MUST reach the
// same energy. The restricted leg wires the RI-J link's fused BuildFock; the
// unrestricted leg wires the per-spin adapter's three calls
// (MakeRiJLinkUhfFockBuilder). That is one Fock matrix through two code
// paths, so the pair is the adapter's correctness pin.
std::string H2oDef2SvpUhfToml(std::string_view builder) {
    std::string text = H2oDef2SvpToml(builder);
    constexpr std::string_view kRhfWord = "type = \"rhf\"";
    const auto methodPos = text.find(kRhfWord);

    if (methodPos == std::string::npos)
    {
        ADD_FAILURE() << "the fixture text lacks the [method] method word";
        return text;
    }

    text.replace(methodPos, kRhfWord.size(), "type = \"uhf\"");
    return text;
}

// The alkane chain TOML of the composed builder chain pins: an n-carbon STO-3G RHF
// text over the shared fixture geometry, the [basis] sto-3g name resolving
// from the compiled-in basis data.
//
// THE SCF GATE IS THE defaults rule DEFAULT (1e-8 energy / 1e-6 density), not the
// 1e-10/1e-10 pair the defaults rule's explicit-gate list hardened these rows to. Owner
// ruling 2026-09-18, on the pins' cost: the standing rule bans a 1e-10
// energy or density threshold without raising it with the owner first, and
// 1e-10/1e-10 is seven orders below the 9.1e-4 band the kLoose rows validate
// against - reference precision the claims never asked for. Measured on
// this tree, MSVC Release, same fixture and preset, at the two gates
// (iterations / SCF wall): C6 direct kTight 15 / 11.4 s vs 12 / 9.0 s;
// C8 direct kTight 12 / 25.5 s; C6 qfmm kTight 12 / 55.3 s; C8 qfmm
// kNormal 17 / 161 s vs 12 / 150 s. NO ASSERTION MOVED WITH THE GATE: the
// rows' bands, their references and their expected values are unchanged,
// and every row is re-measured at the new gate in the tests below.
std::string AlkaneChainToml(const qcx::molecule::Molecule& molecule,
                            std::string_view builder,
                            std::string_view accuracy) {
    std::ostringstream text;
    WriteAlkaneMoleculeBlock(text, molecule);
    text << "\n[basis]\norbital = \"sto-3g\"\n\n[method]\ntype = \"rhf\"\nfock_builder = \""
         << builder << "\"\naccuracy = \"" << accuracy
         << "\"\n\n[scf]\nmax_iterations = 100\n"
            "energy_tolerance = 1e-8\ndensity_tolerance = 1e-6\nuse_diis = true\n";
    return text.str();
}

// The comparison pair: ONE molecule block (H2 at 1.4 bohr -
// the geometry of the driver's own HF pin and of the KsCompositionTest
// fixture), ONE basis, ONE [method].accuracy, and NO [scf] block at all, so
// every run takes the defaults rule defaults (1e-8 energy / 1e-6 density). The only
// text that differs between any two runs is the [method] body the caller
// passes, so whatever separates their energies is the method word and
// nothing else.
std::string H2MethodToml(std::string_view methodBody) {
    return std::string(R"(
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
)") + std::string(methodBody) +
           "accuracy = \"kNormal\"\n";
}

} // namespace

TEST(DriverPinTest, H2DirectRhfReproducesThePin) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kH2Toml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, -1.1167143252, 1e-8);

    const auto iterations = JsonNumber(*result, "\"iterations\"");
    ASSERT_TRUE(iterations.has_value());
    EXPECT_GT(static_cast<int>(*iterations), 0);

    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(result->find("\"spin_squared\": null"), std::string::npos);
    EXPECT_NE(
        result->find("\"schema_version\": " + std::to_string(qcx::io::RunResult::kSchemaVersion)),
        std::string::npos);

    // Every driver run resolves the [resources] block into
    // resources_resolved, even a defaulted one. The defaults must
    // reproduce the pin exactly. in_process_cap_applied is asserted as
    // key-present only: Linux CI has no job objects and honestly reports
    // false there, so the value is platform-dependent by design.
    EXPECT_NE(result->find("\"resources_resolved\""), std::string::npos);
    EXPECT_NE(result->find("\"memory_cap_gib\": 16.0"), std::string::npos);
    EXPECT_NE(result->find("\"thread_cap\": 0"), std::string::npos);
    EXPECT_NE(result->find("\"in_process_cap_applied\""), std::string::npos);

    EXPECT_NE(result->find("\"builder\": \"direct\""), std::string::npos);
    // 's lean flag and 's member name. This run spells
    // fock_builder = "direct" explicitly, so it takes the machinery branch
    // at every size (Ruling B): the flag stays false and the member name
    // is the family word itself - off the lean carve-out the two agree,
    // and the family word carries no lean spelling at all.
    EXPECT_NE(result->find("\"builder_member\": \"direct\""), std::string::npos);

    // The unrestricted coefficients block is a RETIRED PRODUCER: the
    // candidate ranking that filled it died with the lean flip, so no
    // driver path sets `coefficients` and the serializer's
    // branch never fires. A real run therefore carries no file_version or
    // scores member at all - the record's "none" is the block's ABSENCE,
    // never an empty block that reads as a ranking which found nothing.
    // Those two keys are emitted in that block and nowhere else, so their
    // absence pins the claim from the record side: a producer cannot be
    // reintroduced without failing here.
    EXPECT_EQ(result->find("\"file_version\""), std::string::npos);
    EXPECT_EQ(result->find("\"scores\""), std::string::npos);

    // block: the H2 STO-3G pins of populations_test.cpp /
    // multipoles_test.cpp through the full driver path. Each probe anchors
    // on its own member name: "total" alone is ambiguous - the serializer
    // emits the lowdin block before mulliken, so the first "total" after
    // "populations" is lowdin's (caught when the H2O Mulliken/Lowdin pins
    // diverged, 2026-08-25).
    const auto populationsPos = result->find("\"populations\"");
    ASSERT_NE(populationsPos, std::string::npos);
    const auto mullikenPos = result->find("\"mulliken\"", populationsPos);
    ASSERT_NE(mullikenPos, std::string::npos);
    const auto mullikenTotal = JsonNumberArray(*result, "\"total\":", mullikenPos);
    ASSERT_TRUE(mullikenTotal.has_value());
    ASSERT_EQ(mullikenTotal->values.size(), 2u);
    EXPECT_NEAR(mullikenTotal->values[0], 1.0, 1e-7);
    EXPECT_NEAR(mullikenTotal->values[1], 1.0, 1e-7);
    const auto mullikenSpin = JsonNumberArray(*result, "\"spin\":", mullikenPos);
    ASSERT_TRUE(mullikenSpin.has_value());
    ASSERT_EQ(mullikenSpin->values.size(), 2u);
    EXPECT_NEAR(mullikenSpin->values[0], 0.0, 1e-7);
    EXPECT_NEAR(mullikenSpin->values[1], 0.0, 1e-7);
    const auto lowdinPos = result->find("\"lowdin\"");
    ASSERT_NE(lowdinPos, std::string::npos);
    const auto lowdinTotal = JsonNumberArray(*result, "\"total\":", lowdinPos);
    ASSERT_TRUE(lowdinTotal.has_value());
    ASSERT_EQ(lowdinTotal->values.size(), 2u);
    EXPECT_NEAR(lowdinTotal->values[0], 1.0, 1e-7);
    EXPECT_NEAR(lowdinTotal->values[1], 1.0, 1e-7);
    const auto dipole = JsonNumberArray(*result, "\"dipole\":");
    ASSERT_TRUE(dipole.has_value());
    ASSERT_EQ(dipole->values.size(), 3u);
    EXPECT_NEAR(dipole->values[0], 0.0, 1e-7);
    EXPECT_NEAR(dipole->values[1], 0.0, 1e-7);
    EXPECT_NEAR(dipole->values[2], 0.0, 1e-7);
    EXPECT_NE(result->find("\"quadrupole\""), std::string::npos);
    EXPECT_NE(result->find("\"scf_loop\""), std::string::npos);
}

// The ladder's bottom tier, end to end and through the run record
// . The same H2 fixture with no [method].fock_builder key: at
// nBasis 2 <= the 1000-basis-function boundary the driver wires the direct
// family's within-family lean member, and the selection record must
// say so. The family word stays "direct" - the lean member carries no
// builder-slot spelling (the flag is the only record of the choice) - so
// this test and H2DirectRhfReproducesThePin are the flag's true/false
// pair on one fixture.
TEST(DriverPinTest, H2AutoLeanWritesTheLeanMember) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kH2TomlAutoLean);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_NE(
        result->find("\"schema_version\": " + std::to_string(qcx::io::RunResult::kSchemaVersion)),
        std::string::npos);
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(result->find("\"builder\": \"direct\""), std::string::npos);
    EXPECT_NE(result->find("\"builder_member\": \"lean\""), std::string::npos);

    // The lean path is still the pinned H2: the flag records the wiring,
    // never a different physical result.
    const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, -1.1167143252, 1e-8);
}

// The explicit `fock_builder = "lean"` spelling, end to end: the word is a
// within-family request (no BuilderKind), so the record's family word stays
// "direct" and the lean flag carries the choice - exactly the auto-lean
// record's shape (H2AutoLeanWritesTheLeanMember pins the other spelling on
// the same fixture). The resolution's reasoning records the request
// (no family word was given, so the auto text would otherwise claim a bare
// default).
TEST(DriverPinTest, H2ExplicitLeanWordWiresTheLeanMember) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kH2TomlLeanWord);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(result->find("\"builder\": \"direct\""), std::string::npos);
    EXPECT_NE(result->find("\"builder_member\": \"lean\""), std::string::npos);

    // The lean member carries no family word, so the record's
    // explicit_builder stays absent (the null honesty policy) - the
    // reasoning is the record of the request.
    EXPECT_NE(result->find("explicit fock_builder lean honored"), std::string::npos);
    EXPECT_EQ(result->find("\"explicit_builder\""), std::string::npos);

    const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, -1.1167143252, 1e-8);
}

// ---------------------------------------------------------------------------
// the Kohn-Sham words reach the run that composes the XC
// potential into the Fock build. Until this wiring landed the dispatch REFUSED both
// words, and the refusal was the point: the run path selected its method by
// equality, so an unwired `rks`/`uks` did not fail - it ran Hartree-Fock and
// reported the energy under a DFT label, a wrong scientific answer that looks
// exactly like a right one. The composition is what makes the difference
// visible; these tests are the end-to-end half of the evidence, and
// KsCompositionTest's 12 run-layer tests are the seam-side half.
//
// Every run below is the same fixture one [method] body apart - H2/STO-3G at
// 1.4 bohr, the geometry of the driver's own HF pin and of the composition
// fixture, at the defaults rule [scf] defaults (no [scf] block is written). A
// fall-through to the Hartree-Fock branch would reproduce -1.1167143252
// exactly, so the margin is the assertion that the XC potential reached the
// Fock build; the agreement across the two builder families and the two SCF
// loops is what says it reached it correctly.
// ---------------------------------------------------------------------------

// Slater is a pure functional - excgrid's registry carries no exact-exchange
// fraction for it - so the composition builds no K half at all (the refusal
// side is pinned by RksPureFunctionalNeverBuildsTheExchangeHalf) and the
// whole exchange is the engine's own LDA number.
TEST(DriverPinTest, RksSlaterH2RunsTheKohnShamPathOnBothBuilderMembers) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const std::string hfText = H2MethodToml("type = \"rhf\"\n");
    auto hf = RunInputText(hfText.c_str());
    ASSERT_TRUE(hf.has_value()) << hf.error().message;
    const auto hfTotal = JsonNumber(*hf, "\"total_energy_hartree\"");
    ASSERT_TRUE(hfTotal.has_value());
    // The fixture is the pinned one. If this moved, the comparison below
    // would be measuring the geometry rather than the method.
    EXPECT_NEAR(*hfTotal, -1.1167143252, 1e-8);

    // The lean member, asked for BY NAME. The keyless spelling is no longer
    // this member on a pure functional: since the RI-J link's Kohn-Sham
    // composition landed, the ladder's non-hybrid promotion (NamedTier puts
    // ri_j_link above lean when c_HF = 0) resolves to a tier the wiring can
    // now actually run, so the keyless leg wires ri_j_link and is pinned by
    // PureKohnShamOrderRunsThePromotedTier next door. What this test owns is
    // the DIRECT family's two members, so it asks for them.
    const std::string leanText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nfock_builder = \"lean\"\n");
    auto lean = RunInputText(leanText.c_str());
    ASSERT_TRUE(lean.has_value()) << lean.error().message;
    const auto leanTotal = JsonNumber(*lean, "\"total_energy_hartree\"");
    ASSERT_TRUE(leanTotal.has_value());
    EXPECT_NE(lean->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(lean->find("\"builder_member\": \"lean\""), std::string::npos);

    // The machinery member: the explicit family word takes the machinery
    // branch at every size (Ruling B) - a different integral engine behind
    // the same halves.
    const std::string machineryText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nfock_builder = \"direct\"\n");
    auto machinery = RunInputText(machineryText.c_str());
    ASSERT_TRUE(machinery.has_value()) << machinery.error().message;
    const auto machineryTotal = JsonNumber(*machinery, "\"total_energy_hartree\"");
    ASSERT_TRUE(machineryTotal.has_value());
    EXPECT_NE(machinery->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(machinery->find("\"builder_member\": \"direct\""), std::string::npos);

    // NOT Hartree-Fock: Slater exchange moves this fixture's total by
    // 9.17e-2 Ha, eight orders of magnitude above the pin's tolerance.
    EXPECT_GT(std::abs(*leanTotal - *hfTotal), 1e-3);

    // The driver's number IS the composition's own measured RKS energy for
    // this fixture (KsCompositionTest.SlaterRksRunThroughTheRealEngineIsSelfConsistent,
    // which reached it through RunRhfScf and the seam directly). Two entry
    // points, one number - so the driver's wiring is not a second
    // implementation that happens to run.
    EXPECT_NEAR(*leanTotal, -1.025009958088, 1e-9);
    // And the two builder families agree, so the wiring is not one arm
    // agreeing with itself.
    EXPECT_NEAR(*leanTotal, *machineryTotal, 1e-9);
}

// ---------------------------------------------------------------------------
// The grid + the method record: the record names the run's OWN PHYSICS - which method
// ran, which functional produced the numbers, and which grid they were
// integrated on - and the grid becomes a key a file can actually set.
//
// Both halves are asserted end to end here rather than on the serializer
// alone, because the failure each one guards against is a wiring failure and
// not a formatting one: The method record's defect was that a record named its Fock builder
// and no physics (two runs differing by the whole RKS-vs-RHF difference were
// otherwise indistinguishable in the artifact a user keeps), and the grid's was
// that the six grid settings were compile-time constants no input could reach
// and no record could name. A record-only change would leave the second defect
// in place while every key looked right, so the grid key is also shown MOVING
// the energy it was used to compute.
// ---------------------------------------------------------------------------
TEST(DriverPinTest, TheRecordNamesTheMethodTheFunctionalAndTheGridItBuilt) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // The Hartree-Fock arm (the lean member: this fixture's cheapest build).
    // The method word is there, and the two Kohn-Sham payloads are ABSENT -
    // no density functional was named and no XC grid was built, so there is
    // nothing to disclose and nothing to fabricate (`functional` is not null
    // and `xc_grid` is not a block of defaults).
    const std::string hfText = H2MethodToml("type = \"rhf\"\nfock_builder = \"lean\"\n");
    auto hf = RunInputText(hfText.c_str());
    ASSERT_TRUE(hf.has_value()) << hf.error().message;
    EXPECT_NE(hf->find("\"method\": \"rhf\""), std::string::npos);
    EXPECT_EQ(hf->find("\"functional\""), std::string::npos);
    EXPECT_EQ(hf->find("\"xc_grid\""), std::string::npos);

    // The Kohn-Sham arm, the same fixture one method word apart: the record
    // names the method, the functional the registry resolved for the input's
    // name, and the six settings the grid engine was created with.
    const std::string rksText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nfock_builder = \"lean\"\n");
    auto rks = RunInputText(rksText.c_str());
    ASSERT_TRUE(rks.has_value()) << rks.error().message;
    EXPECT_NE(rks->find("\"method\": \"rks\""), std::string::npos);
    EXPECT_NE(rks->find("\"functional\": \"slater\""), std::string::npos);
    const auto rksTotal = JsonNumber(*rks, "\"total_energy_hartree\"");
    ASSERT_TRUE(rksTotal.has_value());

    // No [grid] block was written, so the record carries the engine's OWN
    // defaults - the behaviour every run had before the block existed, now
    // stated rather than implied. The numbers are pinned as literals: this is
    // the disclosure surface, and a default that drifted would silently change
    // the quadrature of every file that never wrote the block.
    EXPECT_NE(rks->find("\"radial_points\": 75"), std::string::npos);
    EXPECT_NE(rks->find("\"angular_points\": 302"), std::string::npos);
    EXPECT_NE(rks->find("\"alpha\": 0.5"), std::string::npos);
    EXPECT_NE(rks->find("\"radial_exponent\": 2"), std::string::npos);
    EXPECT_NE(rks->find("\"trim_weight\": 1e-15"), std::string::npos);
    EXPECT_NE(rks->find("\"block_target\": 1024"), std::string::npos);

    // The grid reaches the BUILD, not just the record: the same run at a
    // coarser quadrature (20 radial points, a 50-point Lebedev set) moves the
    // energy, and the record discloses the coarser numbers. A key that was
    // parsed, disclosed and never handed to XcGridEngine::Create would pass
    // every assertion above this one.
    const std::string coarseText = rksText + "[grid]\nradial_points = 20\nangular_points = 50\n";
    auto coarse = RunInputText(coarseText.c_str());
    ASSERT_TRUE(coarse.has_value()) << coarse.error().message;
    EXPECT_NE(coarse->find("\"radial_points\": 20"), std::string::npos);
    EXPECT_NE(coarse->find("\"angular_points\": 50"), std::string::npos);
    EXPECT_EQ(coarse->find("\"radial_points\": 75"), std::string::npos);
    const auto coarseTotal = JsonNumber(*coarse, "\"total_energy_hartree\"");
    ASSERT_TRUE(coarseTotal.has_value());
    EXPECT_GT(std::abs(*rksTotal - *coarseTotal), 1e-6);
}

// The ladder's ONE exception, end to end (the owner's ruling 2026-09-13): a
// NON-HYBRID Kohn-Sham run resolves against the two-tier order - ri_j_link
// BEFORE the lean member - because a pure functional needs J and no K.
//
// This test used to pin the DEMOTION: the promoted tier existed only in the
// resolver, the v1 wiring had no Kohn-Sham ri_j arm, and the run landed on
// the lean member with a warning naming the tier it could not wire. That gap
// is closed (the composition over RiJkFockBuilder's two halves), so the same
// keyless fixture now RUNS the promoted tier, and the demotion clause is
// gone. The old assertions are not deleted, they are inverted with their
// reason: "names the ri_j_link tier" was the demotion's own text, and its
// absence here is what says the tier was wired rather than announced.
//
// The hybrid counterpart is the key's other side, and it is the half that
// matters most: keying the exception on "is it Kohn-Sham" instead of on the
// functional's character would put every B3LYP/PBE0 run on the two-tier order
// too. Same fixture, one word changed.
TEST(DriverPinTest, PureKohnShamOrderRunsThePromotedTier) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const std::string pureText = H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\n");
    auto pure = RunInputText(pureText.c_str());
    ASSERT_TRUE(pure.has_value()) << pure.error().message;

    // The promoted order, and now what it DID: the clause says why the order
    // was the two-tier one (the record still carries it - the resolver's
    // reasoning is the same for a tier that runs as for one that demotes),
    // and the run below is the promoted tier itself.
    EXPECT_NE(pure->find("NON-HYBRID (pure) Kohn-Sham run"), std::string::npos);
    EXPECT_NE(pure->find("puts ri_j_link BEFORE the lean member"), std::string::npos);
    EXPECT_NE(pure->find("\"builder_member\": \"ri_j_link\""), std::string::npos);
    EXPECT_NE(pure->find("\"converged\": true"), std::string::npos);
    // No demotion was needed, so neither the demotion's reasoning nor its
    // stderr warning is present - the two are produced together at one site.
    EXPECT_EQ(pure->find("names the ri_j_link tier"), std::string::npos);
    EXPECT_EQ(pure->find("demoted to the direct family's lean member"), std::string::npos);
    EXPECT_EQ(pure->find("\"warning\":"), std::string::npos);
    // Nor the lean member's flag: the family word and the member name are
    // both the promoted tier's.

    const std::string hybridText = H2MethodToml("type = \"rks\"\nfunctional = \"b3lyp\"\n");
    auto hybrid = RunInputText(hybridText.c_str());
    ASSERT_TRUE(hybrid.has_value()) << hybrid.error().message;

    // A hybrid needs K, so it keeps the shared order: no clause, no
    // promotion, no demotion - the pre-ruling record byte for byte. And it
    // lands on the LEAN member, not on the promoted tier: the exception is
    // the functional's character and nothing else.
    EXPECT_EQ(hybrid->find("NON-HYBRID"), std::string::npos);
    EXPECT_EQ(hybrid->find("names the ri_j_link tier"), std::string::npos);
    EXPECT_EQ(hybrid->find("\"warning\":"), std::string::npos);
    EXPECT_NE(hybrid->find("\"builder_member\": \"lean\""), std::string::npos);
}

// The RI-J link's Kohn-Sham composition, measured against the direct family
// on one deck - the correctness pin the acceptance asks for.
//
// The comparison is a PURE functional (Slater, c_HF = 0) on purpose: with no
// exact exchange the two wirings differ in exactly ONE operator, the Coulomb
// half, and everything else - the core Hamiltonian, the grid engine, the XC
// potential, the walk - is byte-identical between them. That makes the
// difference between the two totals the RI-J fit's own error and nothing
// else, which is what lets the pin below state what it establishes and what
// it does not:
//
//   - it establishes the promoted tier RUNS and computes the Kohn-Sham Fock
//     the composition describes. A wiring that dropped the Coulomb half's H,
//     or handed the seam a Coulomb matrix that is not the Fock's (a K-folded
//     one, or one built at another density), is wrong by O(H) or O(K) - three
//     to five orders above this margin, not below it.
//   - it does NOT establish that the RI-J approximation is accurate. That is
//     a property of the fitted auxiliary basis, it is shared with every other
//     RI-J run in the tree, and the pin records the number it is on this
//     fixture rather than asserting a small one.
//
// The hybrid half of the same question is RiJLinkKsHybridRunsTheExchangeHalf
// below.
TEST(DriverPinTest, RksRiJLinkMatchesTheDirectFamilyOnTheSameDeck) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // The promoted tier, reached the way a user reaches it: no key at all.
    const std::string promotedText = H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\n");
    auto promoted = RunInputText(promotedText.c_str());
    ASSERT_TRUE(promoted.has_value()) << promoted.error().message;
    EXPECT_NE(promoted->find("\"builder\": \"ri_j_link\""), std::string::npos) << *promoted;
    EXPECT_NE(promoted->find("\"builder_member\": \"ri_j_link\""), std::string::npos);
    EXPECT_NE(promoted->find("\"converged\": true"), std::string::npos);
    const auto promotedTotal = JsonNumber(*promoted, "\"total_energy_hartree\"");
    ASSERT_TRUE(promotedTotal.has_value());

    // The same functional on the direct family's machinery member: the exact
    // Coulomb half, the same everything else.
    const std::string directText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nfock_builder = \"direct\"\n");
    auto direct = RunInputText(directText.c_str());
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    const auto directTotal = JsonNumber(*direct, "\"total_energy_hartree\"");
    ASSERT_TRUE(directTotal.has_value());

    std::printf("RKS slater H2/STO-3G: ri_j_link = %.15e, direct = %.15e, difference = %.15e\n",
                *promotedTotal,
                *directTotal,
                *promotedTotal - *directTotal);

    // MEASURED, on this binary and this fixture: the difference is
    // -6.867078214334299e-05 Ha. It is the RI-J fit's error and nothing else,
    // and the two readings that establish that are both about the
    // DIFFERENCE's own behaviour rather than about its size:
    //
    //   - it moves with the AUXILIARY QUALITY. KsCompositionTest's
    //     self-consistency run fits the same H2/STO-3G fixture on its own
    //     orbital set and lands 1.5e-01 Ha away from the exact-Coulomb
    //     result - four orders further out, on the same geometry. A
    //     structural wiring defect (a missing H, a K folded into the Coulomb
    //     half, a J built at another density) does not move with the fit.
    //   - it does NOT move with the FUNCTIONAL. RiJLinkKsHybridRunsThe
    //     ExchangeHalf asserts the hybrid's difference equals this one.
    //
    // The bound below sits ~3x above the measured value for the arithmetic's
    // own latitude (a different BLAS sums the contraction in another order)
    // and remains four orders BELOW the smallest wiring mistake the checks
    // above would catch.
    EXPECT_NEAR(*promotedTotal, *directTotal, 2e-4);

    // And both are Kohn-Sham, not Hartree-Fock: the fall-through the DFT
    // wiring exists to prevent would land on the pinned RHF number.
    EXPECT_GT(std::abs(*promotedTotal - (-1.1167143252)), 1e-3);
}

// The promoted tier's unrestricted twin, and the cross-loop agreement the
// restricted counterpart cannot check: H2 at multiplicity 1 is the
// closed-shell limit, so RKS and UKS on this fixture are the same physics
// through two different loops and two different density parameterizations.
//
// The UKS composition halves the XC potential differently from the RKS one
// (per spin, no 1/2 - the engine returns dE/dD_s), and its Coulomb half is
// keyed on the spin-summed total while its exchange halves are keyed on the
// pair. Agreement between the two loops here is what says those conventions
// were applied the right way round; a consumer that swapped them would miss
// by a factor of two, not by rounding.
TEST(DriverPinTest, UksRiJLinkMatchesTheRestrictedLimitOnTheSameDeck) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const std::string promotedText = H2MethodToml("type = \"uks\"\nfunctional = \"slater\"\n");
    auto promoted = RunInputText(promotedText.c_str());
    ASSERT_TRUE(promoted.has_value()) << promoted.error().message;
    EXPECT_NE(promoted->find("\"builder\": \"ri_j_link\""), std::string::npos) << *promoted;
    EXPECT_NE(promoted->find("\"builder_member\": \"ri_j_link\""), std::string::npos);
    EXPECT_NE(promoted->find("\"converged\": true"), std::string::npos);
    const auto promotedTotal = JsonNumber(*promoted, "\"total_energy_hartree\"");
    ASSERT_TRUE(promotedTotal.has_value());

    const std::string directText =
        H2MethodToml("type = \"uks\"\nfunctional = \"slater\"\nfock_builder = \"direct\"\n");
    auto direct = RunInputText(directText.c_str());
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    const auto directTotal = JsonNumber(*direct, "\"total_energy_hartree\"");
    ASSERT_TRUE(directTotal.has_value());

    const std::string rksText = H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\n");
    auto rks = RunInputText(rksText.c_str());
    ASSERT_TRUE(rks.has_value()) << rks.error().message;
    const auto rksTotal = JsonNumber(*rks, "\"total_energy_hartree\"");
    ASSERT_TRUE(rksTotal.has_value());

    std::printf("UKS slater H2/STO-3G: ri_j_link = %.15e, direct = %.15e, difference = %.15e\n",
                *promotedTotal,
                *directTotal,
                *promotedTotal - *directTotal);

    // Same agreement as the restricted lane's, on the same fixture: the
    // per-spin composition is the same operator split the other way.
    EXPECT_NEAR(*promotedTotal, *directTotal, 2e-4);
    // The two loops land on the same promoted-tier number, and it is not a
    // near miss - MEASURED bit-identical to the restricted run's on this
    // binary (-1.025078628869946 on both). Same builder, same J, different
    // density parameterization; the bound leaves room for the two loops'
    // own accumulation order rather than for a modelling difference.
    EXPECT_NEAR(*promotedTotal, *rksTotal, 1e-10);
}

// The hybrid half: a functional with c_HF > 0 must build the exchange half as
// well, and the composition must subtract the exchange term it is scaled by.
// Slater cannot reach that code at all (the K half is never created), so a
// promotion-only test above leaves the hybrid path unwitnessed.
//
// Asked for BY NAME rather than left to the ladder: a hybrid functional keeps
// the SHARED order (canonical tier at nBasis 2 is the lean member), so the
// keyless spelling would test the direct family. The key is what puts the
// hybrid's two halves on the RI-J link, and the deck is the same one.
TEST(DriverPinTest, RiJLinkKsHybridRunsTheExchangeHalf) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const std::string riJText =
        H2MethodToml("type = \"rks\"\nfunctional = \"b3lyp\"\nfock_builder = \"ri_j_link\"\n");
    auto riJ = RunInputText(riJText.c_str());
    ASSERT_TRUE(riJ.has_value()) << riJ.error().message;
    EXPECT_NE(riJ->find("\"builder\": \"ri_j_link\""), std::string::npos) << *riJ;
    EXPECT_NE(riJ->find("\"converged\": true"), std::string::npos);
    const auto riJTotal = JsonNumber(*riJ, "\"total_energy_hartree\"");
    ASSERT_TRUE(riJTotal.has_value());

    const std::string directText =
        H2MethodToml("type = \"rks\"\nfunctional = \"b3lyp\"\nfock_builder = \"direct\"\n");
    auto direct = RunInputText(directText.c_str());
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    const auto directTotal = JsonNumber(*direct, "\"total_energy_hartree\"");
    ASSERT_TRUE(directTotal.has_value());

    std::printf("RKS b3lyp H2/STO-3G: ri_j_link = %.15e, direct = %.15e, difference = %.15e\n",
                *riJTotal,
                *directTotal,
                *riJTotal - *directTotal);

    // The same agreement the pure functional gets, because the ONLY operator
    // the two wirings differ in is still the Coulomb half: the exchange half
    // is the nested direct-exchange builder's own result on both legs
    // (RiJkFockBuilder::BuildExchangeOnly IS the direct family's
    // buildExchangeOnly mode), so the hybrid adds a term both sides share
    // exactly and leaves the difference where the pure case put it.
    EXPECT_NEAR(*riJTotal, *directTotal, 2e-4);

    const std::string slaterText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nfock_builder = \"ri_j_link\"\n");
    auto slater = RunInputText(slaterText.c_str());
    ASSERT_TRUE(slater.has_value()) << slater.error().message;
    const auto slaterTotal = JsonNumber(*slater, "\"total_energy_hartree\"");
    ASSERT_TRUE(slaterTotal.has_value());
    // Not the pure-functional number: the exact-exchange fraction moved the
    // total, so a run that silently dropped the K half fails here rather than
    // passing the agreement above for the wrong reason.
    EXPECT_GT(std::abs(*riJTotal - *slaterTotal), 1e-3);

    // THE INVARIANCE, and it is the sharpest statement this deck can make
    // about the difference above. MEASURED: the pure and the hybrid legs land
    // on the SAME difference to 4.4e-16 Ha - i.e. to the arithmetic's own
    // last bit - while their totals differ by 1.3e-01 Ha. Asserted at 1e-12,
    // four orders inside the totals' separation.
    //
    // What that establishes is exactly the thing a tolerance cannot: the
    // difference carries NO component that scales with the exact-exchange
    // fraction. A Coulomb half that had the K term folded into it, or an
    // exchange half reaching the wrong callable, would put a c_HF-scaled
    // term in here and separate the two numbers by ~c_HF |K| - hundreds of
    // micro-hartree for B3LYP's 0.2, not 4e-16.
    //
    // WHAT IT DOES NOT ESTABLISH, stated so the invariance is not read
    // further than it goes: on THIS fixture the density matrix is
    // symmetry-determined (H2 in a minimal basis is one bonding combination
    // whatever the functional), so the RI-J error is the same number on both
    // legs. On a fixture whose density moves with the functional the two
    // differences would differ by the fit's own sensitivity to D - a
    // property of the fit, not of the wiring - and this assertion belongs to
    // this deck alone.
    const std::string slaterDirectText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nfock_builder = \"direct\"\n");
    auto slaterDirect = RunInputText(slaterDirectText.c_str());
    ASSERT_TRUE(slaterDirect.has_value()) << slaterDirect.error().message;
    const auto slaterDirectTotal = JsonNumber(*slaterDirect, "\"total_energy_hartree\"");
    ASSERT_TRUE(slaterDirectTotal.has_value());

    const double hybridDifference = *riJTotal - *directTotal;
    const double pureDifference = *slaterTotal - *slaterDirectTotal;
    std::printf("the difference itself: hybrid = %.15e, pure = %.15e, separation = %.3e\n",
                hybridDifference,
                pureDifference,
                hybridDifference - pureDifference);
    EXPECT_NEAR(hybridDifference, pureDifference, 1e-12);
}

// The composed-QFMM builder's Kohn-Sham composition, measured against the
// direct family on one deck - the correctness pin the acceptance asks for,
// and the third family to reach this pair of words.
//
// The comparison is a PURE functional (Slater, c_HF = 0) on purpose: with no
// exact exchange the two wirings differ in exactly ONE operator, the Coulomb
// half, and everything else - the core Hamiltonian, the grid engine, the XC
// potential, the walk - is byte-identical between them. What that makes the
// difference is not uniform across families, and the distinction is the
// interesting part here: the RI-J link's difference is the fit's own error
// (~1e-4), while the composed QFMM builder's J half on THIS fixture is the
// RESTRICTED DIRECT build itself - the octree's far field is vacuous at two
// basis functions, so no multipole approximation runs and the two wirings
// contract the same operator. The pin below is therefore asserted at the
// arithmetic's own scale, not at an approximation's, and what it establishes
// and what it cannot are both worth stating exactly:
//
//   - it establishes the promoted wiring RUNS and computes the Kohn-Sham
//     Fock the composition describes. A wiring that dropped an H, that added
//     one back where the family's halves already carry it, or that handed
//     the seam a Coulomb matrix that is not the Fock's (a K-folded one, or
//     one built at another density), is wrong by O(H) or O(K) - three to
//     five orders above this margin, not below it.
//   - it does NOT establish anything about the QFMM far field, BECAUSE NONE
//     RAN on this fixture: measured through the CLI on this same deck (the
//     composed-QFMM RHF run's trace rows read far_pairs=0 at kNormal, and
//     the count is a Create-time property of the octree the geometry and
//     QfmmOptions decide - the KS run builds the same one), so the composed
//     builder degenerates to the near-field direct build and the
//     near-exactness above says nothing about the multipole expansion. The
//     far field's approximating behaviour is pinned where it is live, on
//     the larger decks (H2oQfmmComposedRhfReproducesThePinAtAllPresets and
//     the chain pins), and no Kohn-Sham cell of this deck reaches that size.
TEST(DriverPinTest, RksQfmmMatchesTheDirectFamilyOnTheSameDeck) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const std::string qfmmText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nfock_builder = \"qfmm\"\n");
    auto qfmm = RunInputText(qfmmText.c_str());
    ASSERT_TRUE(qfmm.has_value()) << qfmm.error().message;
    EXPECT_NE(qfmm->find("\"builder\": \"qfmm\""), std::string::npos) << *qfmm;
    EXPECT_NE(qfmm->find("\"builder_member\": \"qfmm\""), std::string::npos) << *qfmm;
    EXPECT_NE(qfmm->find("\"converged\": true"), std::string::npos) << *qfmm;
    // The QFMM model the Coulomb half ran: a Kohn-Sham run on
    // this family states it exactly as the Hartree-Fock one does, because the
    // same half ran.
    EXPECT_NE(qfmm->find("\"qfmm_model\""), std::string::npos) << *qfmm;
    const auto qfmmTotal = JsonNumber(*qfmm, "\"total_energy_hartree\"");
    ASSERT_TRUE(qfmmTotal.has_value());

    const std::string directText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nfock_builder = \"direct\"\n");
    auto direct = RunInputText(directText.c_str());
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    const auto directTotal = JsonNumber(*direct, "\"total_energy_hartree\"");
    ASSERT_TRUE(directTotal.has_value());

    std::printf("RKS slater H2/STO-3G: qfmm = %.15e, direct = %.15e, difference = %.15e\n",
                *qfmmTotal,
                *directTotal,
                *qfmmTotal - *directTotal);

    // MEASURED on this binary and this fixture: the two totals are the SAME
    // DOUBLE - the printf above prints their difference as exactly
    // 0.000000000000000e+00, not as a small nonzero. The bound below is the
    // arithmetic's own latitude, and it stays four orders inside the smallest
    // wiring mistake this deck can express (a K folded into the Coulomb half
    // moves this fixture by 1e-2 Ha; a stray H by the core Hamiltonian).
    EXPECT_NEAR(*qfmmTotal, *directTotal, 1e-12)
        << "qfmm " << *qfmmTotal << " direct " << *directTotal;

    // And it is Kohn-Sham, not Hartree-Fock: the fall-through a fused seam
    // would produce lands on the pinned RHF number, three orders away.
    EXPECT_GT(std::abs(*qfmmTotal - (-1.1167143252)), 1e-3);
}

// The same pair on the unrestricted word, plus the cross-loop agreement the
// restricted lane cannot check: H2 at multiplicity 1 is the closed-shell
// limit, so RKS and UKS on this fixture are the same physics through two
// different loops and two different density parameterizations.
//
// The UKS composition halves the XC potential differently from the RKS one
// (per spin, no 1/2 - the engine returns dE/dD_s), and its Coulomb half is
// keyed on the spin-summed total while its exchange halves are keyed on the
// pair. Agreement between the two loops here is what says those conventions
// were applied the right way round; a consumer that swapped them would miss
// by a factor of two, not by rounding.
TEST(DriverPinTest, UksQfmmMatchesTheRestrictedLimitOnTheSameDeck) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const std::string uksText =
        H2MethodToml("type = \"uks\"\nfunctional = \"slater\"\nfock_builder = \"qfmm\"\n");
    auto uks = RunInputText(uksText.c_str());
    ASSERT_TRUE(uks.has_value()) << uks.error().message;
    EXPECT_NE(uks->find("\"builder\": \"qfmm\""), std::string::npos) << *uks;
    EXPECT_NE(uks->find("\"converged\": true"), std::string::npos) << *uks;
    EXPECT_NE(uks->find("\"qfmm_model\""), std::string::npos) << *uks;
    const auto uksTotal = JsonNumber(*uks, "\"total_energy_hartree\"");
    ASSERT_TRUE(uksTotal.has_value());

    const std::string directText =
        H2MethodToml("type = \"uks\"\nfunctional = \"slater\"\nfock_builder = \"direct\"\n");
    auto direct = RunInputText(directText.c_str());
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    const auto directTotal = JsonNumber(*direct, "\"total_energy_hartree\"");
    ASSERT_TRUE(directTotal.has_value());

    const std::string rksText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nfock_builder = \"qfmm\"\n");
    auto rks = RunInputText(rksText.c_str());
    ASSERT_TRUE(rks.has_value()) << rks.error().message;
    const auto rksTotal = JsonNumber(*rks, "\"total_energy_hartree\"");
    ASSERT_TRUE(rksTotal.has_value());

    std::printf("UKS slater H2/STO-3G: qfmm = %.15e, direct = %.15e, difference = %.15e\n",
                *uksTotal,
                *directTotal,
                *uksTotal - *directTotal);
    std::printf("the two loops: uks-qfmm = %.15e, rks-qfmm = %.15e, separation = %.3e\n",
                *uksTotal,
                *rksTotal,
                *uksTotal - *rksTotal);

    // The restricted lane's own agreement, on the per-spin loop: the same
    // operator split the other way. MEASURED on this binary: the difference
    // is exactly 0.0, as the restricted twin's is.
    EXPECT_NEAR(*uksTotal, *directTotal, 1e-12);
    // The two loops land on the same number, and it is not a near miss:
    // MEASURED bit-identical to the restricted run's on this binary (the
    // printf above prints 0.000e+00 for their separation). Same halves,
    // different density parameterization; the bound leaves room for the two
    // loops' own accumulation order rather than for a modelling difference.
    EXPECT_NEAR(*uksTotal, *rksTotal, 1e-10);
}

// The hybrid half on the composed-QFMM builder: a functional with c_HF > 0
// must build the exchange half as well, and the composition must subtract the
// exchange term it is scaled by. Slater cannot reach that code at all (no
// exchange half is passed), so the pins above leave the hybrid path
// unwitnessed.
TEST(DriverPinTest, QfmmKsHybridRunsTheExchangeHalf) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const std::string qfmmText =
        H2MethodToml("type = \"rks\"\nfunctional = \"b3lyp\"\nfock_builder = \"qfmm\"\n");
    auto qfmm = RunInputText(qfmmText.c_str());
    ASSERT_TRUE(qfmm.has_value()) << qfmm.error().message;
    EXPECT_NE(qfmm->find("\"builder\": \"qfmm\""), std::string::npos) << *qfmm;
    EXPECT_NE(qfmm->find("\"converged\": true"), std::string::npos) << *qfmm;
    const auto qfmmTotal = JsonNumber(*qfmm, "\"total_energy_hartree\"");
    ASSERT_TRUE(qfmmTotal.has_value());

    const std::string directText =
        H2MethodToml("type = \"rks\"\nfunctional = \"b3lyp\"\nfock_builder = \"direct\"\n");
    auto direct = RunInputText(directText.c_str());
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    const auto directTotal = JsonNumber(*direct, "\"total_energy_hartree\"");
    ASSERT_TRUE(directTotal.has_value());

    std::printf("RKS b3lyp H2/STO-3G: qfmm = %.15e, direct = %.15e, difference = %.15e\n",
                *qfmmTotal,
                *directTotal,
                *qfmmTotal - *directTotal);

    // The pure lane's agreement, with the exchange half now live on both
    // legs: the K half is the nested exchange-only direct builder's own
    // result on both, so the hybrid adds a term both sides share exactly and
    // leaves the difference where the pure case put it.
    EXPECT_NEAR(*qfmmTotal, *directTotal, 1e-12);

    const std::string slaterText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nfock_builder = \"qfmm\"\n");
    auto slater = RunInputText(slaterText.c_str());
    ASSERT_TRUE(slater.has_value()) << slater.error().message;
    const auto slaterTotal = JsonNumber(*slater, "\"total_energy_hartree\"");
    ASSERT_TRUE(slaterTotal.has_value());
    // Not the pure-functional number: the exact-exchange fraction moved the
    // total, so a run that silently dropped the K half fails here rather than
    // passing the agreement above for the wrong reason.
    EXPECT_GT(std::abs(*qfmmTotal - *slaterTotal), 1e-3);

    // THE INVARIANCE, and it is the sharpest statement this deck can make
    // about the difference above. MEASURED: the pure and the hybrid legs land
    // on the SAME difference - both exactly 0.0 on this binary, asserted at
    // 1e-13 below - while their totals differ by 1.34e-01 Ha (measured:
    // -1.158599927859593 against -1.025009958087802).
    //
    // What that establishes is exactly the thing a tolerance cannot: the
    // difference carries NO component that scales with the exact-exchange
    // fraction. A Coulomb half that had the K term folded into it, an
    // exchange half reaching the wrong callable, or an H added back at the
    // half (the RI-J link's accounting imported into a family whose halves
    // already carry theirs), would put a c_HF-scaled term in here: the
    // hybrid's difference would leave zero while the pure one stayed, and
    // the assertion below would fire on the ~c_HF |K| separation - hundreds
    // of micro-hartree for B3LYP's 0.2, not 1e-13.
    //
    // DEGENERATE ON THIS DECK, said because it bounds the reading: both
    // differences are exactly 0.0, so "the two are equal" is satisfied by two
    // zeros and discriminates only what leaves the zero - i.e. exactly the
    // c_HF-scaled terms above. On the RI-J link's deck the difference is the
    // fit's own ~1e-4 and the same assertion is sharp in a second way (it
    // also excludes a term that moves with the density the functional
    // chooses); here the QFMM half is the same exact operator on both legs,
    // so there is no such second reading to make.
    //
    // WHAT IT DOES NOT ESTABLISH, stated so the invariance is not read
    // further than it goes: on THIS fixture the density matrix is
    // symmetry-determined (H2 in a minimal basis is one bonding combination
    // whatever the functional) AND the QFMM far field is vacuous, so both
    // legs evaluate the same exact Coulomb operator at the same density. On a
    // fixture with a live far field the two differences would differ by the
    // multipole's own sensitivity to D - a property of the approximation, not
    // of the wiring - and this assertion belongs to this deck alone.
    const std::string slaterDirectText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nfock_builder = \"direct\"\n");
    auto slaterDirect = RunInputText(slaterDirectText.c_str());
    ASSERT_TRUE(slaterDirect.has_value()) << slaterDirect.error().message;
    const auto slaterDirectTotal = JsonNumber(*slaterDirect, "\"total_energy_hartree\"");
    ASSERT_TRUE(slaterDirectTotal.has_value());

    const double hybridDifference = *qfmmTotal - *directTotal;
    const double pureDifference = *slaterTotal - *slaterDirectTotal;
    std::printf("the difference itself: hybrid = %.15e, pure = %.15e, separation = %.3e\n",
                hybridDifference,
                pureDifference,
                hybridDifference - pureDifference);
    EXPECT_NEAR(hybridDifference, pureDifference, 1e-13);
}

// The cells this wiring did NOT open, each refused BY NAME rather than run.
// Two separate reasons, and the refusals have to name the right one - a
// reader who gets "needs its own composition" for a builder that does expose
// halves would look in the wrong place:
//   - gpu / gpu_split on a Kohn-Sham run: the combination check's whitelist
//     (gpu fuses J and K into one kernel launch and would need two calls for a
//     composition);
//   - the ri_j_link disk rungs on a Kohn-Sham run: the storage-module disk
//     builder exposes no half either, and the ladder's rung word and the
//     diagnostic force are two keys onto that one cell - both refused, so the
//     request cannot be routed around by naming the other.
//
// qfmm LEFT this list on 2026-09-16 (the composition over the composed
// builder's two exposed halves, QfmmHfFockBuilder::BuildCoulombOnly /
// BuildExchangeOnly): it was the third family in this loop, and the run it
// now reaches is pinned by RksQfmmMatchesTheDirectFamilyOnTheSameDeck and
// its unrestricted twin below. ri_jk LEFT it the same day and the same way
// (the composition over RiFullFockBuilder::BuildFockHalves, whose bare halves
// the driver's MakeRiFullKsHalf converts: H into both, the factor of two onto
// the Coulomb one) - the run it now reaches is pinned by
// RksRiJkRunsTheComposedFullRiCompositionAndDisclosesIt below. The loop keeps
// the ONE family that still refuses, and the ordering check below is stated on
// it so it cannot be lost with a family that moved.
TEST(DriverErrorTest, KohnShamRefusesTheFamiliesAndRungsOutsideTheComposition) {
    // The refused message, or an empty string plus a recorded failure when the
    // run came back - the caller's EXPECT_NE on "" then reads as "the refusal
    // did not fire", which is the failure that matters here.
    const auto refusedMessage = [](const std::string& text) {
        auto result = RunInputText(text.c_str());

        if (result.has_value())
        {
            ADD_FAILURE() << "expected a refusal, got a run whose record starts: "
                          << result->substr(0, 200);
            return std::string{};
        }

        return result.error().message;
    };

    for (const char* family : {"gpu"})
    {
        const std::string rksText = H2MethodToml(std::string("type = \"rks\"\n"
                                                             "functional = \"slater\"\n"
                                                             "fock_builder = \"") +
                                                 family + "\"\n");
        const std::string message = refusedMessage(rksText);
        EXPECT_NE(message.find("Kohn-Sham"), std::string::npos) << message;
        EXPECT_NE(message.find(std::string("\"") + family + "\""), std::string::npos) << message;
        // The reason is this family's own: the device builder fuses J and K
        // into one batched pass, so a composition would need two calls of the
        // whole cost. The sentence an ri_jk reader used to get here ("the pair
        // is BARE") must NOT ride this refusal: that family's halves ARE
        // exposed and the driver converts them now, so a refusal that named
        // their accounting would send its reader after a mechanism their
        // family already has - the same drift this assertion caught when qfmm
        // was the family that moved.
        EXPECT_NE(message.find("fuses its Coulomb and exchange halves inside one BuildFock"),
                  std::string::npos)
            << message;
        EXPECT_EQ(message.find("the pair is BARE"), std::string::npos)
            << "the lifted ri_jk reason rode this refusal: " << message;
        // The admitted set is named, so the refusal is actionable - and it is
        // the set the wiring really admits, qfmm, ri_j_link and ri_jk included.
        EXPECT_NE(message.find("fock_builder = \"direct\", \"qfmm\", \"ri_j_link\" and "
                               "\"ri_jk\""),
                  std::string::npos)
            << message;

        // The ORDERING, which is the reason this loop runs on the
        // UNRESTRICTED Kohn-Sham word: a UKS request is both unrestricted and
        // Kohn-Sham, so whichever whitelist ValidateCombination reads first
        // decides which reason it gets. The Kohn-Sham rules are read first
        // (the block's own note), so the refusal must be about the
        // composition - NOT the unrestricted leg's "no per-spin adapter",
        // which would send a UKS author after a mechanism their run was never
        // going to use. An insertion that moved the unrestricted rule above
        // the Kohn-Sham one would leave the family named and the word
        // "Kohn-Sham" in place while this assertion went red.
        //
        // The SENDER is asserted as well, and on this family it is what tells
        // the two refusals apart: BOTH whitelists refuse gpu, so an inverted
        // ordering would answer a UKS request with the unrestricted leg's "no
        // per-spin adapter" - a refusal about a mechanism a Kohn-Sham run was
        // never going to use. "is wired with" is the combination rule's own
        // phrase and the "no per-spin adapter" clause is the other rule's, so
        // the two assertions below separate "refused up front" from "refused
        // by the leg that would not have served it anyway". (The family this
        // paragraph was written for - ri_jk, which the unrestricted leg ADMITS
        // - no longer refuses on either rule, so the older form of the check,
        // where only one of the two texts could appear at all, has nothing
        // left to separate.)
        const std::string uksText = H2MethodToml(std::string("type = \"uks\"\n"
                                                             "functional = \"slater\"\n"
                                                             "fock_builder = \"") +
                                                 family + "\"\n");
        const std::string uksMessage = refusedMessage(uksText);
        EXPECT_NE(uksMessage.find("Kohn-Sham"), std::string::npos) << uksMessage;
        EXPECT_NE(uksMessage.find("is wired with"), std::string::npos)
            << "the refusal did not come from the combination rule (an inverted ordering would "
               "answer with the unrestricted leg's own text instead): "
            << uksMessage;
        EXPECT_EQ(uksMessage.find("no per-spin adapter"), std::string::npos) << uksMessage;
    }

    // The disk rungs, both spellings, on both Kohn-Sham lanes. The reason
    // names the disk builder's own gap - never the unrestricted leg's
    // missing per-spin adapter, which would send a Kohn-Sham reader after a
    // mechanism their run never had.
    for (const char* method : {"rks", "uks"})
    {
        const std::string rungText = H2MethodToml(std::string("type = \"") + method +
                                                  "\"\nfunctional = \"slater\"\n"
                                                  "fock_builder = \"ri_j_link\"\n"
                                                  "ri_tensor_mode = \"disk\"\n");
        const std::string rungMessage = refusedMessage(rungText);
        EXPECT_NE(rungMessage.find("no J[D] for its energy seam"), std::string::npos)
            << rungMessage;
        EXPECT_EQ(rungMessage.find("no per-spin adapter"), std::string::npos) << rungMessage;

        // The forced diagnostic force: the other key onto the same cell, and
        // the one the ladder never sees (so it cannot be demoted away - it
        // reaches the wiring and is refused there).
        const std::string forcedText = H2MethodToml(std::string("type = \"") + method +
                                                    "\"\nfunctional = \"slater\"\n"
                                                    "fock_builder = \"ri_j_link\"\n") +
                                       "\n[diagnostics]\nforce_disk_ri = true\n";
        EXPECT_NE(refusedMessage(forcedText).find("no J[D] for its energy seam"),
                  std::string::npos);
    }
}

// The per-spin loop on the same fixture. H2 with multiplicity 1 is the
// closed-shell limit, so UKS must land on the RKS number: a different loop, a
// different density parameterization (two spin densities through the engine's
// pair entry point), the same physics. Agreement here is what says the
// per-spin wiring halves the right quantity - the engine returns dE/dD_s, so
// a consumer that dropped the restricted path's 1/2, or applied it on the
// unrestricted lane, would miss by a factor rather than by noise.
//
// Both legs name the DIRECT family explicitly: the keyless spelling of a pure
// functional is the promoted ri_j_link tier (PureKohnShamOrderRunsThePromotedTier),
// and an RI-J Coulomb half is a different operator from this family's exact
// one - comparing across the two would fold the fit's error into a check
// whose whole point is to be exact. The promoted tier's own cross-loop
// agreement is pinned separately, on the same fixture.
TEST(DriverPinTest, UksSlaterH2MatchesTheRestrictedLimitOnBothBuilderMembers) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const std::string rksText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nfock_builder = \"lean\"\n");
    auto rks = RunInputText(rksText.c_str());
    ASSERT_TRUE(rks.has_value()) << rks.error().message;
    const auto rksTotal = JsonNumber(*rks, "\"total_energy_hartree\"");
    ASSERT_TRUE(rksTotal.has_value());

    const std::string leanText =
        H2MethodToml("type = \"uks\"\nfunctional = \"slater\"\nfock_builder = \"lean\"\n");
    auto lean = RunInputText(leanText.c_str());
    ASSERT_TRUE(lean.has_value()) << lean.error().message;
    const auto leanTotal = JsonNumber(*lean, "\"total_energy_hartree\"");
    ASSERT_TRUE(leanTotal.has_value());
    EXPECT_NE(lean->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(lean->find("\"builder_member\": \"lean\""), std::string::npos);

    const std::string machineryText =
        H2MethodToml("type = \"uks\"\nfunctional = \"slater\"\nfock_builder = \"direct\"\n");
    auto machinery = RunInputText(machineryText.c_str());
    ASSERT_TRUE(machinery.has_value()) << machinery.error().message;
    const auto machineryTotal = JsonNumber(*machinery, "\"total_energy_hartree\"");
    ASSERT_TRUE(machineryTotal.has_value());
    EXPECT_NE(machinery->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(machinery->find("\"builder_member\": \"direct\""), std::string::npos);

    // The unrestricted lane's own pin. An unrestricted run is free to break
    // spin symmetry; on this fixture it does not, and the number is the
    // restricted one.
    EXPECT_NEAR(*leanTotal, -1.025009958088, 1e-9);
    EXPECT_NEAR(*leanTotal, *machineryTotal, 1e-9);
    EXPECT_NEAR(*rksTotal, *leanTotal, 1e-9);

    // Not Hartree-Fock either. RHF and UHF agree on this closed-shell
    // fixture, so a UKS that fell through to the UHF branch would land on
    // the same -1.1167143252 the restricted fall-through would.
    EXPECT_GT(std::abs(*leanTotal - (-1.1167143252)), 1e-3);
}

// The two grid tolerances give the same number here, and that is a
// measurement rather than an assumption: the composition's screen is exact at
// this size (a two-function basis, one shell pair), so io's default 1e-10 and
// the dense 0.0 spelling agree digit for digit. Worth pinning, because the
// alternative - a default that quietly moved the energy - would leave every
// comparison above resting on a tolerance nobody chose.
//
// Both legs are the KEYLESS spelling, so both are the ladder's promoted
// tier now (ri_j_link) rather than the lean member this test's first
// version ran. That does not weaken it - the key is the screen, and the two
// legs still differ in nothing else - but it does move what the row covers:
// on this route the tolerance reaches the XC grid engine's evaluator
// (ResolveKsContext passes it to EvaluateScreened) and not a direct-exchange
// screen, so the assertion is that the grid screen does not move the
// energy. Named here so a reader does not take it for a statement about
// Schwarz screening on the direct family.
TEST(DriverPinTest, RksScreeningToleranceDoesNotMoveTheH2Energy) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const std::string denseText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nscreening_tolerance = 0.0\n");
    auto dense = RunInputText(denseText.c_str());
    ASSERT_TRUE(dense.has_value()) << dense.error().message;
    const auto denseTotal = JsonNumber(*dense, "\"total_energy_hartree\"");
    ASSERT_TRUE(denseTotal.has_value());

    const std::string screenedText =
        H2MethodToml("type = \"rks\"\nfunctional = \"slater\"\nscreening_tolerance = 1e-10\n");
    auto screened = RunInputText(screenedText.c_str());
    ASSERT_TRUE(screened.has_value()) << screened.error().message;
    const auto screenedTotal = JsonNumber(*screened, "\"total_energy_hartree\"");
    ASSERT_TRUE(screenedTotal.has_value());

    EXPECT_NEAR(*denseTotal, *screenedTotal, 1e-10);
}

// thread_cap = 1 clamps the team to one thread, so ChunkCountFor yields
// numChunks = 1 - the single-chunk serial path, whose bit-identity is
// pinned. The equivalence proof: two identical serial-path runs are
// bit-for-bit equal (single-thread accumulation has no merge-order drift
// to differ), and the serial result sits within the documented
// combine-order tolerance of the parallel pin. The ceiling is process-
// wide, so it is restored before any assertion - a failure here can never
// poison the rest of the suite (and the serial path being within pin
// tolerance of the parallel pins is exactly the equivalence being pinned,
// so even a poisoned suite would stay green).
TEST(DriverPinTest, ThreadCapOneDegradesToTheD59SerialPath) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    qcx::backend::SetOmpThreadCeiling(1);

    auto first = RunInputText(kH2TomlThreadCap1);
    auto second = RunInputText(kH2TomlThreadCap1);

    qcx::backend::SetOmpThreadCeiling(0);

    ASSERT_TRUE(first.has_value()) << first.error().message;
    ASSERT_TRUE(second.has_value()) << second.error().message;

    // The single-chunk serial-path bit-identity pin, at the driver level:
    // the same thread-cap-1 input run twice produces the same energy bits
    // (single-thread accumulation has no merge-order drift). The full
    // documents are NOT comparable - timings_ms is wall clock.
    const auto firstTotal = JsonNumber(*first, "\"total_energy_hartree\"");
    const auto secondTotal = JsonNumber(*second, "\"total_energy_hartree\"");
    ASSERT_TRUE(firstTotal.has_value());
    ASSERT_TRUE(secondTotal.has_value());
    EXPECT_EQ(*firstTotal, *secondTotal)
        << "the serial path (numChunks 1) must be deterministic bit-for-bit";
    EXPECT_NEAR(*firstTotal, -1.1167143252, 1e-8);

    EXPECT_NE(first->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(first->find("\"thread_cap\": 1"), std::string::npos);
}

TEST(DriverPinTest, DensityAtNucleiBlockEndToEnd) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // end to end: [properties] density_at_nuclei = true through
    // the full driver path. The pin is the pyscf 2.14.0 row at the
    // kH2Toml geometry - molecule coordinates are in Angstrom, so the
    // 0.7408480953 A bond is the 1.4 bohr fixture, and the pyscf row
    // used unit='Bohr' (the standing Angstrom-default trap):
    // rho = 0.354892073081 at each nucleus.
    const auto toml = std::string(kH2Toml) + "[properties]\ndensity_at_nuclei = true\n";
    auto result = RunInputText(toml.c_str());
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_NE(result->find("\"density_at_nuclei\""), std::string::npos);
    const auto values = JsonNumberArray(*result, "\"values\"");
    ASSERT_TRUE(values.has_value()) << "missing density_at_nuclei.values";
    ASSERT_EQ(values->values.size(), 2u);
    EXPECT_NEAR(values->values[0], 0.354892073081, 1e-8);
    EXPECT_NEAR(values->values[1], 0.354892073081, 1e-8);
}

TEST(DriverPinTest, H2QtaimEndToEnd) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // end to end: [properties] qtaim = true through the full
    // driver path. H2/STO-3G has exactly one (3,-1) bond critical point at
    // the bond midpoint (0.7 bohr; the kH2Toml bond is 0.7408480953 A =
    // 1.4 bohr) with the qtaim_test pins (pyscf 2.14.0 rows). The bond
    // path terminates at both nuclei - no otherCriticalPoints, no
    // unconvergedSeeds.
    const auto toml = std::string(kH2Toml) + "[properties]\nqtaim = true\n";
    auto result = RunInputText(toml.c_str());
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_NE(result->find("\"qtaim\""), std::string::npos);
    std::size_t bcpCount = 0;
    std::size_t searchPos = 0;

    while ((searchPos = result->find("\"atom_a\":", searchPos)) != std::string::npos)
    {
        ++bcpCount;
        searchPos += 9;
    }

    EXPECT_EQ(bcpCount, 1u) << "exactly one bond critical point for H2";
    EXPECT_NE(result->find("\"atom_a\": 0"), std::string::npos);
    EXPECT_NE(result->find("\"atom_b\": 1"), std::string::npos);

    const auto position = JsonNumberArray(*result, "\"position_bohr\":");
    ASSERT_TRUE(position.has_value()) << "missing bond_critical_points.position_bohr";
    EXPECT_EQ(position->values.size(), 3u);
    EXPECT_NEAR(position->values[0], 0.7, 1e-6);
    EXPECT_NEAR(position->values[1], 0.0, 1e-6);
    EXPECT_NEAR(position->values[2], 0.0, 1e-6);

    const auto density = JsonScalarNumber(*result, "\"density\":");
    ASSERT_TRUE(density.has_value()) << "missing bond_critical_points.density";
    EXPECT_NEAR(*density, 0.255926671299, 1e-8);

    // The laplacian pin class is 1e-7 (qtaim_test.cpp): the Hessian trace
    // carries the finite-difference curvature noise in its last digits
    // (libcgto ships no GTOval_ipip - qtaim_test.cpp:50-53), unlike the
    // density row's tight 1e-8 agreement.
    const auto laplacian = JsonScalarNumber(*result, "\"laplacian\":");
    ASSERT_TRUE(laplacian.has_value()) << "missing bond_critical_points.laplacian";
    EXPECT_NEAR(*laplacian, -0.840815721791, 1e-7);

    // Axial symmetry: the two negative eigenvalues are exactly equal along
    // the bond, so the ellipticity is zero at this geometry.
    const auto ellipticity = JsonScalarNumber(*result, "\"ellipticity\":");
    ASSERT_TRUE(ellipticity.has_value()) << "missing bond_critical_points.ellipticity";
    EXPECT_NEAR(*ellipticity, 0.0, 1e-8);

    const auto eigenvalues = JsonNumberArray(*result, "\"eigenvalues\":");
    ASSERT_TRUE(eigenvalues.has_value()) << "missing bond_critical_points.eigenvalues";
    ASSERT_EQ(eigenvalues->values.size(), 3u);
    EXPECT_LT(eigenvalues->values[0], 0.0);
    EXPECT_LT(eigenvalues->values[1], 0.0);
    EXPECT_GT(eigenvalues->values[2], 0.0);
    EXPECT_LE(eigenvalues->values[0], eigenvalues->values[1]);
    EXPECT_LE(eigenvalues->values[1], eigenvalues->values[2]);

    EXPECT_NE(result->find("\"bond_path\""), std::string::npos);
    EXPECT_NE(result->find("\"other_critical_points\": []"), std::string::npos);
    EXPECT_NE(result->find("\"unconverged_seeds\": []"), std::string::npos);

    // The unrequested block is absent from the document (the absent-not-null
    // contract), never null.
    auto plain = RunInputText(kH2Toml);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    EXPECT_EQ(plain->find("\"qtaim\""), std::string::npos);
}

TEST(DriverPinTest, RhfGwhGuessMatchesTheCoreDefault) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // guess gwh for RHF is wired through the restart seam and equals
    // the loop's own default start (the GWH guess), so the explicit gwh
    // request must reproduce the core-default pin bit for bit - both take
    // the guess default start. This guards the driver seam, not the SCF (the loop
    // default is pinned by rhf_convergence_test.cpp).
    auto result = RunInputText(R"(
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

[guess]
type = "gwh"
)");
    // Measured 2026-09-16 and VESTIGIAL: this fixture is kH2Toml with the guess
    // seam named, and the cell's subject is the seam - not the gate. At the
    // operating default it reproduces -1.116714325175768 (2.42e-11 from the
    // pinned -1.1167143252, band 1e-8) in 2 iterations, converged true: the same
    // record the 1e-10 pair gave.
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, -1.1167143252, 1e-8);

    const auto iterations = JsonNumber(*result, "\"iterations\"");
    ASSERT_TRUE(iterations.has_value());
    EXPECT_GT(static_cast<int>(*iterations), 0);
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);
}

TEST(DriverPinTest, H2oDirectRhfReproducesThePin) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kH2oToml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, -74.96292827, 1e-5);
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

    // through the driver path: the H2O dipole (multipoles_test
    // H2oSto3gDipoleAndQuadrupole, same geometry) and the oxygen Mulliken
    // population (populations_test H2oSto3gPins; anchored on "mulliken" -
    // see the H2 test's comment on the serializer's member order).
    const auto dipole = JsonNumberArray(*result, "\"dipole\":");
    ASSERT_TRUE(dipole.has_value());
    ASSERT_EQ(dipole->values.size(), 3u);
    EXPECT_NEAR(dipole->values[1], 0.67898079210, 1e-7);
    const auto populationsPos = result->find("\"populations\"");
    ASSERT_NE(populationsPos, std::string::npos);
    const auto mullikenPos = result->find("\"mulliken\"", populationsPos);
    ASSERT_NE(mullikenPos, std::string::npos);
    const auto mullikenTotal = JsonNumberArray(*result, "\"total\":", mullikenPos);
    ASSERT_TRUE(mullikenTotal.has_value());
    ASSERT_EQ(mullikenTotal->values.size(), 3u);
    EXPECT_NEAR(mullikenTotal->values[2], 8.3663559645, 1e-7);
}

// The certified-bound observability: the machinery
// builder computes the fp32 lane's accumulated density-weighted
// kernel-bound sum on every call, and the driver now records it instead
// of passing nullptr for it. The three tests below are the block's
// truth table end to end. FALSIFIABILITY: A and B fail if the plumbing is
// reverted - MakeRhfFockBuilder handing the builder `nullptr` instead of
// &callBound leaves the accumulator empty, Calls() stays 0, and the
// serializer writes no certified_bound key, so the block-presence and
// calls>0 assertions go red. C is not falsifiable that way (absence is
// its expected state) and stands as the honesty pin.
//
// A's and B's numerical readings are the fp32 LANE's, and since the compute profile the
// lane's default is a probe's measured ratio rather than a switch - so A and
// B assert the reading that goes with the state this box's own probe resolves
// (CertifiedBoundLaneExpectedOn) instead of assuming the lane is on. The
// block's presence and call count are lane-independent and stay
// unconditional.
//
// The certified fp32 lane's expected state for this process (the
// owner's ruling 2026-09-12; the host arm since): the driver probes
// BOTH compute profiles at the run boundary and seeds the machinery options
// from the one that matches the family the run wired - the host profile for a
// CPU family, the device profile for a GPU one. Every fixture in this family
// runs the direct (CPU) family, so the seed here is the HOST probe's measured
// ratio (DetectHostComputeProfile, run_driver.cpp) against
// kCertifiedLaneMinRatio, and the pins read that same verdict rather than
// assuming a fixed lane state: a machine whose host probe answers above the
// threshold turns the lane ON (an explicit request would win over it, but no
// fixture in this family makes one).
//
// The host probe is NOT behind the device's GpuProbeFloorGiB cap gate - it
// allocates nothing and initializes nothing, so a cap cannot distort it, and
// the cap premise this helper used to assert is gone with that gate.
bool CertifiedBoundLaneExpectedOn() {
    return qcx::integrals::CertifiedLaneDefaultForRatio(
        qcx::backend::DetectHostComputeProfile().fp32ToFp64Ratio);
}

// The certified lane's DECIDED record: the compute
// profile its default was resolved against, plus the verdict and the
// threshold it was compared against. It is emitted on every run that resolved
// a probe, whatever the verdict - a passing gate still emits its readings -
// so this fixture asserts the block even though the
// lane resolves OFF on this box.
//
// FALSIFIABILITY. The run wires the direct family (its lean member), which
// runs on the CPU, so the recorded source must be "host": if the seam ever
// handed the CPU wiring the device probe the key would read "device" and the
// ratio would be the device profile's unknown 1.0 rather than this box's
// measured ~2, and the source assertion goes red. The verdict is checked
// against the rule applied to the block's OWN ratio, read out of the same
// block, so a hard-wired verdict cannot satisfy it either.
//
// SCOPE: THE HOST ARM ONLY. The block this pins is filled by four
// `gpuTarget ?` ternaries (run_driver.cpp, the record block), so it has a host
// half and a device half, and this fixture - like every other test in this
// tree - exercises the host half. The device half (source "device", the
// device's `measured` criterion, and its three not-reported readings) is NOT
// covered here, and cannot be from this suite: GpuFamily answers true only for
// kGpu/kGpuSplit, a non-CUDA build refuses kGpu by name before the record is
// reached, and the device-less fallback resolves an explicit "gpu" request to a CPU
// kind first (DriverErrorTest, GpuWithoutCudaFallsBackToTheHeuristicPick,
// which is also the evidence for this paragraph). The io-side pair pins the
// SERIALIZATION of a device-shaped block, not this producer's device branch;
// run_driver.cpp's record comment names the same gap. The `measured`/`pairs`
// pair below is the one device-versus-host distinction a host run can still
// falsify.
TEST(DriverPinTest, TheComputeProfileRecordCarriesTheHostProbeVerdict) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kH2TomlAutoLean);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

    const auto blockPos = result->find("\"compute_profile\"");
    ASSERT_NE(blockPos, std::string::npos)
        << "a run that resolved a probe must record the profile its certified lane default was "
           "read off, not just the verdict";

    EXPECT_NE(result->find("\"source\": \"host\"", blockPos), std::string::npos)
        << "this fixture runs the direct (CPU) family, so the host arm is the one that counts";

    const auto ratio = JsonNumber(*result, "\"ratio\":", blockPos);
    const auto threshold = JsonNumber(*result, "\"certified_lane_min_ratio\":", blockPos);
    const auto pairs = JsonNumber(*result, "\"pairs\":", blockPos);
    ASSERT_TRUE(ratio.has_value());
    ASSERT_TRUE(threshold.has_value());
    ASSERT_TRUE(pairs.has_value());
    EXPECT_GT(*ratio, 0.0);

    // The recorded verdict IS the rule applied to the recorded ratio.
    const std::string expectedKey = qcx::integrals::CertifiedLaneDefaultForRatio(*ratio)
                                        ? "\"certified_lane_default\": true"
                                        : "\"certified_lane_default\": false";
    EXPECT_NE(result->find(expectedKey, blockPos), std::string::npos);

    // The threshold is the one policy constant, carried rather than copied.
    EXPECT_DOUBLE_EQ(*threshold, qcx::integrals::kCertifiedLaneMinRatio);

    // A MEASURED host reading carries its timed pair count, and the device arm
    // reports none (`pairs` stays 0 there): a record filled from the other arm
    // would read measured true with pairs 0. Skipped when the probe did not
    // measure - the fallback profile is measured false, exactly as the device
    // arm's unmeasured case is, so the condition is on the record's own key
    // rather than on the machine.
    if (result->find("\"measured\": true", blockPos) != std::string::npos)
    {
        EXPECT_GT(*pairs, 0) << "a measured HOST reading carries its timed pairs; the device arm "
                                "reports none, so 0 here means the block was filled from the "
                                "other arm of the four ternaries";
    }
}

TEST(DriverPinTest, CertifiedBoundReportsTheMachineryLaneAtKNormal) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // Butane/STO-3G: the smallest alkane whose kNormal run routes fp32
    // quartets (see CertifiedBoundAlkaneToml on why not water).
    auto molecule = qcx::testing::MakeAlkaneSto3g(4);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto result = RunInputText(CertifiedBoundAlkaneToml(*molecule).c_str());
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

    // The block is present: the machinery member's BuildFock carried the
    // certifiedBoundSumOut out-parameter on every main-SCF call.
    EXPECT_NE(result->find("\"certified_bound\""), std::string::npos);

    const auto calls = JsonNumber(*result, "\"calls\":");
    ASSERT_TRUE(calls.has_value());
    EXPECT_GT(static_cast<int>(*calls), 0);

    // The delivered bound is a real quantity, not a placeholder, WHEN the
    // lane is on: the density-weighted per-quartet bounds the fp32 lane
    // actually delivered sum to something strictly positive. The lane's
    // state is this run's own probe verdict (the compute profile,
    // CertifiedBoundLaneExpectedOn), so the arm below is the reading that
    // goes with the state the box resolves - on a CUDA-less build the lane
    // is off by construction and the block reads its observed zero, which
    // is the other half of the contract.
    const auto maxCall = JsonNumber(*result, "\"max_call_ha\":");
    ASSERT_TRUE(maxCall.has_value());
    const auto lastCall = JsonNumber(*result, "\"last_call_ha\":");
    ASSERT_TRUE(lastCall.has_value());

    if (CertifiedBoundLaneExpectedOn())
    {
        // Butane/STO-3G at kNormal routes fp32 quartets with the lane on
        // (the lane's own probe, 2026-09-11: methane 0, butane 17056), so
        // a strictly positive sum is the value the lane delivered - a 0.0
        // here would mean the routing went dead.
        EXPECT_GT(*maxCall, 0.0);
    } else
    {
        // The lane is off, so no quartet reached the fp32 path and the
        // delivered sum is exactly 0.0 over an empty set - the same
        // observation the explicit lane-off build makes in integrals
        // (CertifiedLaneDeviceDefaultTest: bit-identical bound sums). It
        // is a present, observed zero, never an absent key. The chain
        // below forces last == 0.0 with it.
        EXPECT_DOUBLE_EQ(*maxCall, 0.0);
    }

    // The last call's bound is the final Fock build's own - it cannot
    // exceed the maximum over the run, and neither can be negative.
    EXPECT_GE(*lastCall, 0.0);
    EXPECT_LE(*lastCall, *maxCall);
}

TEST(DriverPinTest, CertifiedBoundIsATrueZeroWhenTheLaneRoutesNothing) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // Two ways the machinery lane delivers exactly nothing, both of which
    // must read as an OBSERVED zero rather than an absent key: kTight,
    // whose routing threshold MixedPrecisionThreshold(kTight) is 0.0 (the
    // gate admits no quartet by construction), and water/STO-3G at
    // kNormal, whose pairs are all too compact for the a-priori gate to
    // clear the 1e-10 budget. The distinction is what keeps a "0.0" from
    // being read as a certification that the run is exact.
    for (const char* const toml : {kH2oToml, kH2oNormalToml})
    {
        auto result = RunInputText(toml);
        ASSERT_TRUE(result.has_value()) << result.error().message;
        EXPECT_NE(result->find("\"certified_bound\""), std::string::npos);

        const auto calls = JsonNumber(*result, "\"calls\":");
        ASSERT_TRUE(calls.has_value());
        EXPECT_GT(static_cast<int>(*calls), 0);
        EXPECT_NE(result->find("\"last_call_ha\": 0.0"), std::string::npos);
        EXPECT_NE(result->find("\"max_call_ha\": 0.0"), std::string::npos);
    }
}

TEST(DriverPinTest, CertifiedBoundIsAbsentOnTheLeanMember) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // The lean member's Schwarz-only builder has no fp32 lane and no
    // bound out-parameter at all, so the run has no such quantity. The
    // key must be ABSENT - a fabricated 0.0 would read as "certified with
    // zero error" on the one path that cannot certify anything.
    auto result = RunInputText(kH2TomlAutoLean);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);
    EXPECT_EQ(result->find("certified_bound"), std::string::npos);
}

// The fourth case of the block's truth table: the ri_j_link route. Its
// BuildFock takes the two-argument form, so the seam hands it no bound
// pointer at all - while the link's nested direct-exchange half runs a LIVE
// fp32 lane at every preset but kTight (ri_engine.cpp:
// useCertifiedMixedPrecision = options.accuracy != kTight) and drops the
// bound that nested call computed. The route therefore has no certified
// quantity to report and the key must be ABSENT. FALSIFIABILITY: while the
// seam observed outside its `requires` branch, this run published
// { calls: N, last_call_ha: 0.0, max_call_ha: 0.0 } - the accumulator
// folded the never-written out-parameter's 0.0 initializer, exactly the
// fabricated certified zero the schema forbids, and this test goes red.
// The machinery leg on the SAME fixture is the instrument check (the key
// the assertion hunts for is one this file can find).
TEST(DriverPinTest, CertifiedBoundIsAbsentOnTheRiJLink) {
    if (!qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "the universal-J aux f shells exceed this build's kMaxEngineL "
                        "(CI lmax=2)";
    }

    auto link = RunInputText(H2oDef2SvpToml("ri_j_link").c_str());
    ASSERT_TRUE(link.has_value()) << link.error().message;
    EXPECT_NE(link->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(link->find("\"builder\": \"ri_j_link\""), std::string::npos);
    EXPECT_EQ(link->find("certified_bound"), std::string::npos);

    auto machinery = RunInputText(H2oDef2SvpToml("direct").c_str());
    ASSERT_TRUE(machinery.has_value()) << machinery.error().message;
    EXPECT_NE(machinery->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(machinery->find("\"certified_bound\""), std::string::npos);
}

// The ri_jk run path end to end - the composed full-RI builder, and its
// acceptance criterion: a run record whose
// selection.builder is "ri_jk". The request is EXPLICIT, so it resolves
// verbatim (the 2026-09-13 size ladder governs the ABSENT key only), wires the
// composed full-RI builder, and reports both the family word and the member
// word the family word cannot state (the ruling: the contraction form, not the
// family restated).
//
// The fixture carries no [basis].aux, so this cell also pins that the wiring
// resolves the aux for the JK KIND: def2-svp maps to def2-universal-jkfit here
// and to def2-universal-jfit for ri_j_link. A wiring that passed kDefault would
// never reach a run - the standing quality ruling refuses it - so the run existing at all is part
// of what this row proves.
//
// FALSIFIABILITY: before the full-RI builder landed the request was refused by name at the top of
// ValidateCombination and this row failed on ASSERT_TRUE. The member assertion
// goes red if BuilderMemberName loses its kRiJk arm, because the record would
// fall back to echoing the family word - the drift ruling exists to prevent.
TEST(DriverPinTest, RiJKWiresTheComposedFullRiBuilderEndToEnd) {
    if (!qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "the JK-fit aux f shells exceed this build's kMaxEngineL (CI lmax=2)";
    }

    const auto riJk = RunInputText(H2oDef2SvpToml("ri_jk").c_str());

    ASSERT_TRUE(riJk.has_value()) << riJk.error().message;
    EXPECT_NE(riJk->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(riJk->find("\"builder\": \"ri_jk\""), std::string::npos);
    EXPECT_NE(riJk->find("\"builder_member\": \"occ_ri_k\""), std::string::npos);
    EXPECT_NE(riJk->find("\"explicit_builder\": \"ri_jk\""), std::string::npos);

    // The approximation disclosure, which is the OTHER half of what
    // removing the refusal owes: the run states that its exchange half is
    // contracted through an auxiliary fit, and names the aux it used. The aux
    // is the auto-selected JK fit - the fixture carries no [basis].aux - and
    // the record names it as the aux IN EFFECT, which is a statement about the
    // run rather than a claim that the input requested it.
    EXPECT_NE(riJk->find("\"approximation\""), std::string::npos);
    EXPECT_NE(riJk->find("\"exchange\": \"occ_ri_k\""), std::string::npos);
    EXPECT_NE(riJk->find("\"aux_basis\": \"def2-universal-jkfit\""), std::string::npos);

    // The MEASURED-error disclosure: the operative half of the
    // rule that governs this feature, which is that it closes with its error
    // DISCLOSED - and a disclosure the record cannot emit is not one. The run
    // states the path's achieved per-atom error BESIDE the bar it is read
    // against, so an energy this path produced can be sized from the JSON
    // alone.
    //
    // Two numbers of different KINDS, and the assertions keep them apart: the
    // bar is THIS run's (the preset's per-atom budget, here kLoose - named by
    // its own word because the record echoes [method] accuracy nowhere else),
    // while the measured value is the PATH's worst characterized cell and
    // carries the fixture and the fit it was taken on. The expected values are
    // read from their own homes (the integrals mapping) rather than typed here,
    // so a record that drifted from the mapping would fail rather than agree
    // with a copy.
    EXPECT_NE(riJk->find("\"exchange_error\""), std::string::npos) << *riJk;

    const auto measuredPerAtom = JsonNumber(*riJk, "\"measured_per_atom_hartree\"");
    const auto barPerAtom = JsonNumber(*riJk, "\"bar_per_atom_hartree\"");
    ASSERT_TRUE(measuredPerAtom.has_value()) << *riJk;
    ASSERT_TRUE(barPerAtom.has_value()) << *riJk;
    EXPECT_DOUBLE_EQ(*measuredPerAtom,
                     qcx::integrals::RiExchangeWorstMeasuredPerAtomError().perAtomEh);
    EXPECT_DOUBLE_EQ(
        *barPerAtom,
        qcx::integrals::RiExchangeErrorBudgetPerAtom(qcx::integrals::AccuracyPreset::kLoose));
    EXPECT_NE(riJk->find("\"measured_on\": \"h2o_sto3g\""), std::string::npos);
    EXPECT_NE(riJk->find("\"measured_aux_basis\": \"def2-universal-jkfit\""), std::string::npos);
    EXPECT_NE(riJk->find("\"bar_preset\": \"kLoose\""), std::string::npos);

    // The absence arm, on the same fixture: an exact run must NOT carry the key,
    // or "present" would stop meaning "approximated" and the disclosure would be
    // a constant rather than a statement.
    const auto exact = RunInputText(H2oDef2SvpToml("direct").c_str());
    ASSERT_TRUE(exact.has_value()) << exact.error().message;
    EXPECT_EQ(exact->find("\"approximation\""), std::string::npos);
    EXPECT_EQ(exact->find("\"exchange_error\""), std::string::npos);

    // The arm the field's own presence rule exists for, and the one a reader is
    // most likely to get wrong: `ri_j_link` DOES approximate - its Coulomb half
    // is fitted through an auxiliary basis - but its exchange half is the exact
    // direct kernel, so the fitted-exchange error is no error of this run's and
    // must not appear on it. That is why the assertions below are the absence of
    // the measured-error key on an exact exchange, and here the absence of the
    // whole block too: def2-svp is in NO weak region, so this run owes no aux
    // notice either, and the block's presence rule (approximate builder OR aux
    // notice) is unsatisfied on both halves. The notice-on-an-exact-kernel arm
    //  describes is pinned by
    // `TheAuxWeakRegionNoticeReachesTheRunRecord` below, which uses a fixture
    // that IS in a weak region - the two cells are the two sides of the rule.
    const auto link = RunInputText(H2oDef2SvpToml("ri_j_link").c_str());
    ASSERT_TRUE(link.has_value()) << link.error().message;
    EXPECT_NE(link->find("\"builder\": \"ri_j_link\""), std::string::npos) << *link;
    EXPECT_EQ(link->find("\"approximation\""), std::string::npos) << *link;
    EXPECT_EQ(link->find("\"exchange_error\""), std::string::npos) << *link;
    EXPECT_EQ(link->find("\"aux_notice\""), std::string::npos) << *link;
}

// The aux-auto rule weak-region aux notice AT THE RUN PATH: the disclosure
// half of the demotion the aux rule performs. The rule always resolves a
// default, so a run whose selection lands in a region the RI fit serves less
// reliably runs anyway - and the record is the only place a reader can learn
// that, because nothing else in the JSON says why the fit was chosen.
//
// The defect this cell closes, named: until this wiring landed, `AuxSelectionNotice` had
// no caller anywhere in the tree except its own unit cells (17 call sites, all
// in integrals/tests/aux_basis_test.cpp), so the `aux_notice` key the io side
// already had a home, a writer and a presence rule for could not appear in a
// real run. A disclosure a run cannot emit is not a disclosure.
//
// The fixture is STO-3G: a minimal basis is the region the aux-auto rule measurement
// actually characterizes (6.99x the def2-svp control, the strongest residual in
// the aux probe), and it is a weak region for BOTH aux consumers,
// so one molecule serves the exact-kernel arm and the fitted one.
//
// FALSIFIABILITY, which is this cell's whole point: make the driver drop the
// notice - delete the `runSelection.approximation->auxNotice = *auxNotice;`
// assignment, or return nullopt from AuxNoticeInEffect - and every `aux_notice`
// assertion below goes red. Measured, not asserted: the mutation was applied to
// the wiring and this cell failed on both arms while the rest of the file
// stayed green (nothing else pins the notice on a real run). The
// `exchange_error` ABSENCE on the ri_j_link arm is the second half: the block
// now exists on an exact kernel there, and the measured-error key must not ride
// it, or a reader would take the ri_jk path's error for this run's.
TEST(DriverPinTest, TheAuxWeakRegionNoticeReachesTheRunRecord) {
    // Arm 1 - the exact-kernel notice block, which is the arm that
    // keeps the RI-J weak regions visible: ri_j_link's exchange half is the
    // exact direct kernel, so the run builds no approximation of its own and the
    // block exists ONLY because the aux selection warns.
    const auto link =
        RunInputText(H2MethodToml("type = \"rhf\"\nfock_builder = \"ri_j_link\"\n").c_str());

    ASSERT_TRUE(link.has_value()) << link.error().message;
    EXPECT_NE(link->find("\"converged\": true"), std::string::npos) << *link;
    EXPECT_NE(link->find("\"approximation\""), std::string::npos) << *link;
    EXPECT_NE(link->find("\"exchange\": \"exact\""), std::string::npos) << *link;
    EXPECT_NE(link->find("\"aux_basis\": \"def2-universal-jfit\""), std::string::npos) << *link;
    EXPECT_NE(link->find("\"aux_notice\""), std::string::npos) << *link;
    EXPECT_NE(link->find("minimal basis"), std::string::npos) << *link;
    EXPECT_EQ(link->find("\"exchange_error\""), std::string::npos) << *link;

    // Arm 2 - the fitted half, and the reason the notice is read at the
    // resolution point rather than at one builder's arm: the same fixture on the
    // full-RI path owes TWO regions, the minimal basis and the unmatched JK
    // request, and a notice that only ever named one of them would be a
    // different disclosure on this arm than on the one above.
    if (qcx::integrals::SupportsL(3))
    {
        const auto riJk =
            RunInputText(H2MethodToml("type = \"rhf\"\nfock_builder = \"ri_jk\"\n").c_str());

        ASSERT_TRUE(riJk.has_value()) << riJk.error().message;
        EXPECT_NE(riJk->find("\"exchange\": \"occ_ri_k\""), std::string::npos) << *riJk;
        EXPECT_NE(riJk->find("\"aux_basis\": \"def2-universal-jkfit\""), std::string::npos)
            << *riJk;
        EXPECT_NE(riJk->find("minimal basis"), std::string::npos) << *riJk;
        EXPECT_NE(riJk->find("RI-JK with no matched JK fit"), std::string::npos) << *riJk;
        // The measured error still rides THIS arm, which is what keeps the two
        // arms distinguishable rather than both reading as "warned".
        EXPECT_NE(riJk->find("\"exchange_error\""), std::string::npos) << *riJk;
    }

    // Arm 3 - the request side, and the half of the rule that is NOT a
    // demotion. An explicit [basis].aux is an enforced request the run USED, so
    // no policy chose it and there is nothing to disclose: disclosure
    // belongs to a demotion, and the notice's own text says that the selection
    // RESOLVED to a fit - a statement about the AUTO rule, which is not the fit
    // this run used.
    //
    // The aux named here is deliberately one the auto rule does NOT pick
    // (`cc-pvtz-jkfit` where sto-3g + ri_jk auto-selects `def2-universal-jkfit`),
    // and the arm is chosen as the ri_jk one on purpose: that arm carries the
    // block whatever the notice does, so `aux_basis` is present to be read and
    // the contradiction is visible in ONE record rather than inferred from an
    // absence. Ungate the emitter and this run carries the notice's own "the
    // selection resolved to def2-universal-jkfit" beside an `aux_basis` reading
    // cc-pvtz-jkfit - two different aux names for one block, the false-record
    // defect in the place a reader looks for the truth. The last assertion is
    // the one that catches it: the auto rule's name must appear NOWHERE here.
    if (qcx::integrals::SupportsL(3))
    {
        const auto plantedJk = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["H", 0.0, 0.0, 0.0],
    ["H", 0.74084809526419992, 0.0, 0.0],
]

[basis]
orbital = "sto-3g"
aux = "cc-pvtz-jkfit"

[method]
type = "rhf"
fock_builder = "ri_jk"
accuracy = "kNormal"
)");

        ASSERT_TRUE(plantedJk.has_value()) << plantedJk.error().message;
        EXPECT_NE(plantedJk->find("\"aux_basis\": \"cc-pvtz-jkfit\""), std::string::npos)
            << *plantedJk;
        EXPECT_EQ(plantedJk->find("\"aux_notice\""), std::string::npos) << *plantedJk;
        // Key-qualified on purpose: `def2-universal-jkfit` is legitimately in
        // this record as `measured_aux_basis` - the fit the PATH's measured error
        // was taken on (`RunExchangeError`, a fact about the ri_jk family's own
        // accuracy cell and not about this run's choice). What must not appear is
        // the auto rule's name presented as THIS run's aux, which is the name the
        // ungated notice text would have carried.
        EXPECT_EQ(plantedJk->find("\"aux_basis\": \"def2-universal-jkfit\""), std::string::npos)
            << *plantedJk;
    }

    // Arm 4 - the same request rule on the DEFAULT RI-J path, where the block has
    // no approximate builder to hang on. The gate is the same one, and it is
    // asserted on the whole block rather than on the notice alone: this run was
    // NOT demoted, so it owes no disclosure at all, and the auto rule's name
    // must not appear here either (an ungated emitter would build the block and
    // put that name in it).
    //
    // NOT this wiring's, named because this arm is where a reader would look for
    // it: on a non-weak `ri_j_link` run the record names NO aux at all - the aux
    // in effect lives only inside the approximation block, and this arm has none
    // - so the record was the only trace of which fit ran. That
    // is a separate gap from the notice's missing producer.
    const auto planted = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["H", 0.0, 0.0, 0.0],
    ["H", 0.74084809526419992, 0.0, 0.0],
]

[basis]
orbital = "sto-3g"
aux = "def2-universal-jkfit"

[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
)");

    ASSERT_TRUE(planted.has_value()) << planted.error().message;
    EXPECT_NE(planted->find("\"converged\": true"), std::string::npos) << *planted;
    EXPECT_EQ(planted->find("\"approximation\""), std::string::npos) << *planted;
    EXPECT_EQ(planted->find("\"aux_notice\""), std::string::npos) << *planted;
    EXPECT_EQ(planted->find("def2-universal-jfit"), std::string::npos) << *planted;
}

// The standing quality ruling at the run path: a JK request whose auxiliary basis IN EFFECT is
// not a JK-optimized fit is refused BY NAME, and the refusal is the
// predicate's own message (integrals JkOptimizedAuxRefusal), so the static
// validation and the wiring cannot drift apart.
//
// Both DOORS into the same defect are covered here, because the predicate is
// over the resolved NAME and that is exactly what makes it see both: the
// auto-selection's door (every cc-*/aug-cc-* name maps to a -rifit, a J-only
// fit - the case an orbital-PREFIX test would have to hard-code today's vendor
// listing to catch) and the explicit [basis].aux door (a -rifit named beside a
// def2-* orbital base, which never touches the orbital prefix at all).
//
// FALSIFIABILITY: drop the IsJkOptimizedAux call from the wiring and the first
// cell runs to convergence instead of refusing - the J-fit path is numerically
// legal and produces an energy, which is precisely why the refusal has to be
// explicit. Drop it from the STATIC validation and the second cell's message
// changes from the aux-quality refusal to the later one, which the message
// assertions catch.
TEST(DriverPinTest, RiJkRefusesAnAuxThatIsNotAJkFit) {
    // The standing quality ruling at the RUN path. It has exactly ONE reachable door since the
    // aux-auto rule
    // ("the aux basis rule becomes a quality tier", commit 5b21a295, landed
    // with this wiring): the aux auto-selection now answers a JK request
    // with a JK fit for every family, so the auto door cannot produce a name the
    // predicate refuses. What remains is the explicit [basis].aux door - a
    // J-only name forced beside a def2-* orbital base, whose own kRiJk mapping
    // is the JK fit and whose prefix therefore says nothing about the override.
    // That is the door an orbital-PREFIX test could not see, and the reason the
    // predicate is over the RESOLVED name.
    //
    // FALSIFIABILITY: drop the IsJkOptimizedAux call from the wiring and this
    // cell runs to convergence instead of refusing - the J-fit path is
    // numerically legal and produces an energy, which is exactly why the
    // refusal has to be explicit rather than implied by the fit resolving.
    const auto viaExplicit = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "def2-svp"
aux = "def2-universal-jfit"

[method]
type = "rhf"
fock_builder = "ri_jk"
accuracy = "kLoose"
)");

    ASSERT_FALSE(viaExplicit.has_value());
    EXPECT_EQ(viaExplicit.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(viaExplicit.error().message.find("requires a JK-optimized auxiliary basis"),
              std::string::npos)
        << viaExplicit.error().message;
    EXPECT_NE(viaExplicit.error().message.find("def2-universal-jfit"), std::string::npos)
        << viaExplicit.error().message;

    // The control that makes the cell attributable to the AUX rather than to
    // ri_jk being broken: the SAME orbital base with the JK fit named explicitly
    // runs. The fixture's own auto-selection resolves to exactly this name, so
    // the pair is the honest A/B.
    //
    // The control is the leg that CARRIES the f shells: both refusals above are
    // decided on the resolved NAME, before any engine is constructed, so they
    // hold at every lmax - the run below does not, and is gated on the engine's
    // coverage (the pattern the sibling RiJKWiresTheComposedFullRiBuilderEnd-
    // ToEnd takes a cell up, and the sibling control inside RiJkCarriesThe-
    // ReductionRequest takes a leg). CI lmax=2 cannot host it.
    if (qcx::integrals::SupportsL(3))
    {
        const auto control = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "def2-svp"
aux = "def2-universal-jkfit"

[method]
type = "rhf"
fock_builder = "ri_jk"
accuracy = "kLoose"
)");

        ASSERT_TRUE(control.has_value()) << control.error().message;
        EXPECT_NE(control->find("\"builder\": \"ri_jk\""), std::string::npos);
    }

    // The SAME door on the UNRESTRICTED leg, whose runner reaches the rule at
    // its own wiring site (RunRiJkUhfScf, `IsJkOptimizedAux` - the rule is not
    // in the combination check, so each leg that resolves an aux owns its
    // check). Nothing else covers that copy: the restricted cells above and
    // the cap-child fixtures are all RHF, so a UHF arm that dropped the call
    // would run a J-only fit under an ri_jk record with every other cell
    // green. FALSIFIABILITY: delete the call from the runner and this cell
    // converges instead of refusing.
    const auto viaExplicitUhf = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "def2-svp"
aux = "def2-universal-jfit"

[method]
type = "uhf"
fock_builder = "ri_jk"
accuracy = "kLoose"
)");

    ASSERT_FALSE(viaExplicitUhf.has_value());
    EXPECT_EQ(viaExplicitUhf.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(viaExplicitUhf.error().message.find("requires a JK-optimized auxiliary basis"),
              std::string::npos)
        << viaExplicitUhf.error().message;
    EXPECT_NE(viaExplicitUhf.error().message.find("def2-universal-jfit"), std::string::npos)
        << viaExplicitUhf.error().message;
}

// The budget enforcement end to end: the [method]
// enforce_certified_bound key reaches the builder's options through the
// driver's wiring, the comparison decides the routing, and the record
// reports both the numbers and the verdict. FALSIFIABILITY: if the driver
// drops the options field (directOptions.enforceCertifiedBoundBudget is
// never set), the block reads `enforced: false` with the enforcement's
// zero defaults and the fell-back/present assertions go red; if the
// builder computes the comparison but does not act on it, routed_quartets
// stays positive against a budget it does not fit.
TEST(DriverPinTest, CertifiedBoundEnforcementDecidesTheRouting) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto molecule = qcx::testing::MakeAlkaneSto3g(4);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // The unenforced sibling first: the same text minus the key, so the
    // enforcement's own members are read against a run that did not ask
    // for them. Its own routed_quartets reads 0.0 in BOTH lane states -
    // the member is filled by the comparison, which an unenforced call
    // never runs (fock_build.cpp gates the whole body on the option) - so
    // it is the key's routing that this assertion reads, not the lane's.
    auto plain = RunInputText(CertifiedBoundAlkaneToml(*molecule).c_str());
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    const auto plainFp32 = JsonNumber(*plain, "\"routed_quartets\":");
    ASSERT_TRUE(plainFp32.has_value());
    EXPECT_DOUBLE_EQ(*plainFp32, 0.0);
    EXPECT_NE(plain->find("\"enforced\": false"), std::string::npos);

    auto enforced = RunInputText(CertifiedBoundEnforcedAlkaneToml(*molecule).c_str());
    ASSERT_TRUE(enforced.has_value()) << enforced.error().message;
    EXPECT_NE(enforced->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(enforced->find("\"enforced\": true"), std::string::npos);

    // The comparison: the routed sum, the preset-derived budget, and the
    // verdict the builder took on it. These are the LANE's numbers - the
    // comparison runs only for a live lane (fock_build.cpp) - so the arm
    // below is chosen by the same probe verdict the delivery pins read.
    const auto budget = JsonNumber(*enforced, "\"budget_ha\":");
    const auto routed = JsonNumber(*enforced, "\"routed_ha\":");
    const auto refusedQuartets = JsonNumber(*enforced, "\"routed_quartets\":");
    ASSERT_TRUE(budget.has_value());
    ASSERT_TRUE(routed.has_value());
    ASSERT_TRUE(refusedQuartets.has_value());

    if (CertifiedBoundLaneExpectedOn())
    {
        EXPECT_GT(*budget, 0.0);
        EXPECT_GT(*routed, *budget);

        // The act, in the record: the lane was refused, so it delivered
        // nothing - and `fell_back_to_fp64` is what says so rather than
        // leaving a bare zero the reader would have to interpret.
        // `routed_quartets` keeps the admissions the comparison ran on
        // (the count the budget refused), so it stays positive here; the
        // delivered bound is the zero that moved.
        EXPECT_NE(enforced->find("\"fell_back_to_fp64\": true"), std::string::npos);
        EXPECT_GT(*refusedQuartets, 0.0);
    } else
    {
        // The lane is off, so the comparison ran vacuously - the contract
        // fock_build.cpp states for a lane that is "off anyway": the key is
        // taken (`enforced` is true above) and every measured member below
        // keeps its observed-zero default, because a build that routed no
        // quartet has nothing to compare. No fall-back is claimed either:
        // nothing was routed, so nothing failed to fit the budget, and a
        // `fell_back_to_fp64: true` here would be a refusal the run never
        // made.
        EXPECT_DOUBLE_EQ(*budget, 0.0);
        EXPECT_DOUBLE_EQ(*routed, 0.0);
        EXPECT_DOUBLE_EQ(*refusedQuartets, 0.0);
        EXPECT_NE(enforced->find("\"fell_back_to_fp64\": false"), std::string::npos);
    }

    // The delivered bound is the zero that moved in BOTH states: the lane
    // on refuses and delivers nothing, the lane off never had anything to
    // deliver.
    EXPECT_NE(enforced->find("\"last_call_ha\": 0.0"), std::string::npos);

    // The behaviour change is the lane's contribution and nothing else:
    // the enforced run lands on the fp64-only answer within the SCF's own
    // convergence scale, and its energy stays on the unenforced run's
    // scale. (The fall-back is the lane-disabled build; the delivered
    // deviation it removes is measured at ~1.7e-9 on this fixture family
    // by certified_budget_test.cpp. Read at this fixture's SCF stopping
    // tolerance, not tighter: the driver converges at 1e-8/1e-6. With the
    // lane off both runs ARE the fp64 build, so the equality is tighter
    // than the case the tolerance is sized for.)
    const auto plainEnergy = JsonNumber(*plain, "\"total_energy_hartree\"");
    const auto enforcedEnergy = JsonNumber(*enforced, "\"total_energy_hartree\"");
    ASSERT_TRUE(plainEnergy.has_value());
    ASSERT_TRUE(enforcedEnergy.has_value());
    EXPECT_NEAR(*enforcedEnergy, *plainEnergy, 1e-8);
}

// The certified fp32 lane's FORCE-ON input (the owner's ruling 2026-09-13):
// [method] force_certified_lane = true.
//
// WHY THIS PIN EXISTS. Until the key landed, both arms of the certified lane's
// default had only ever been exercised by injecting a probe's return value in
// SOURCE - the host-ratio probe forced DetectHostComputeProfile to 31.2, the
// device arm did the same under the device compute profile - so on a machine whose probe resolves
// OFF (this laptop: the host probe measures ~2 against the 4.0 threshold) the
// lane's ON path was unreachable from any input file, and every pin in this
// family could only ever assert the OFF behaviour. This test IS that path:
// the same butane/STO-3G machinery fixture the sibling pins run, one key
// heavier, with the lane's delivered bound asserted STRICTLY POSITIVE and no
// probe-dependent arm anywhere in it.
//
// FALSIFIABILITY. The forced leg and the unforced leg below differ in exactly
// one key, so on a box whose probe resolves OFF the unforced leg reads the
// observed zero that the forced leg's positive bound refutes: if the driver
// accepted the key and dropped it - the silent class this repo refuses by name
// - the forced leg's max_call_ha would read 0.0 here and this test goes red.
// On a box whose probe resolves ON, the mechanism is asserted by the record
// instead (certified_lane_forced, which the same run carries), so the pin is
// meaningful on both kinds of machine rather than passing vacuously on one.
TEST(DriverPinTest, ForcedCertifiedLaneEngagesThroughTheInputAlone) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto molecule = qcx::testing::MakeAlkaneSto3g(4);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto forced =
        RunInputText(CertifiedBoundAlkaneToml(*molecule, "force_certified_lane = true\n").c_str());
    ASSERT_TRUE(forced.has_value()) << forced.error().message;
    EXPECT_NE(forced->find("\"converged\": true"), std::string::npos);

    // The disclosure, in the block a reader looks at for the lane's state: the
    // run says the lane was FORCED rather than probed, and the machine's own
    // reading is still published beside it - the two are never collapsed into
    // one field, because the block's verdict alone (certified_lane_default,
    // false on this box) would read as a measurement of a machine that never
    // ran.
    const auto forcedBlock = forced->find("\"compute_profile\"");
    ASSERT_NE(forcedBlock, std::string::npos);
    EXPECT_NE(forced->find("\"certified_lane_forced\": true", forcedBlock), std::string::npos)
        << "a forced run must record that it was forced";
    EXPECT_NE(forced->find("\"source\": \"host\"", forcedBlock), std::string::npos);

    // The delivered record: a LIVE lane, whatever the probe measured.
    const auto calls = JsonNumber(*forced, "\"calls\":");
    const auto maxCall = JsonNumber(*forced, "\"max_call_ha\":");
    ASSERT_TRUE(calls.has_value());
    ASSERT_TRUE(maxCall.has_value());
    EXPECT_GT(static_cast<int>(*calls), 0);
    EXPECT_GT(*maxCall, 0.0)
        << "force_certified_lane = true must route quartets through the fp32 lane - a 0.0 here "
           "is the lane-off reading, i.e. the key was accepted and dropped";

    // The unforced sibling, one key lighter, is the machine's own verdict -
    // and on a box whose probe resolves OFF it is the zero the forced leg just
    // moved. Asserted as a PAIR so a fixture that stopped routing fp32
    // quartets on both legs cannot pass this test by reading 0.0 twice.
    auto probed = RunInputText(CertifiedBoundAlkaneToml(*molecule).c_str());
    ASSERT_TRUE(probed.has_value()) << probed.error().message;
    const auto probedBlock = probed->find("\"compute_profile\"");
    ASSERT_NE(probedBlock, std::string::npos);
    EXPECT_NE(probed->find("\"certified_lane_forced\": false", probedBlock), std::string::npos)
        << "the unflagged run must say it was not forced, or the flag carries no information";

    const auto probedMax = JsonNumber(*probed, "\"max_call_ha\":");
    ASSERT_TRUE(probedMax.has_value());

    if (!CertifiedBoundLaneExpectedOn())
    {
        EXPECT_DOUBLE_EQ(*probedMax, 0.0)
            << "this box's probe resolves OFF, so the unforced run must read the OFF arm - "
               "otherwise the pair below proves nothing";
        EXPECT_GT(*maxCall, *probedMax)
            << "the forcing key is the only difference between the two runs";
    }
}

// The other half of the ruling: THE PROBE STILL DECIDES THE DEFAULT. The
// force-on key is an override, not a replacement - with the key absent the
// lane's state is the probe's measured verdict on this machine, and the record
// still carries the reading that verdict was read off. Asserted by relating
// the two fields the run itself emits (the recorded ratio, the recorded
// verdict, the delivered bound) rather than by assuming a fixed lane state, so
// the pin holds on a machine that resolves either way.
//
// FALSIFIABILITY. A driver that let the flag leak into the default would turn
// every run's lane on (or off) and break `delivered > 0` ⟺ `default` below on
// one of the two kinds of machine; a driver that recorded the flag from the
// probe instead of from the request would fail the control leg, where the same
// key is added and the flag must flip while the machine's own reading stays
// exactly as it was.
TEST(DriverPinTest, TheProbeDecidesTheLaneDefaultWhenNothingIsForced) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto molecule = qcx::testing::MakeAlkaneSto3g(4);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto probed = RunInputText(CertifiedBoundAlkaneToml(*molecule).c_str());
    ASSERT_TRUE(probed.has_value()) << probed.error().message;
    EXPECT_NE(probed->find("\"converged\": true"), std::string::npos);

    const auto blockPos = probed->find("\"compute_profile\"");
    ASSERT_NE(blockPos, std::string::npos);
    EXPECT_NE(probed->find("\"certified_lane_forced\": false", blockPos), std::string::npos);

    const auto ratio = JsonNumber(*probed, "\"ratio\":", blockPos);
    const auto maxCall = JsonNumber(*probed, "\"max_call_ha\":");
    ASSERT_TRUE(ratio.has_value());
    ASSERT_TRUE(maxCall.has_value());

    // The verdict IS the rule applied to the run's own recorded ratio, and the
    // delivered bound follows it: the lane routes quartets exactly when the
    // probe said it should. (max_call_ha is 0.0 on the OFF arm over an empty
    // set - an observed zero, never an absent key.)
    const bool expectedOn = qcx::integrals::CertifiedLaneDefaultForRatio(*ratio);
    const std::string expectedKey =
        expectedOn ? "\"certified_lane_default\": true" : "\"certified_lane_default\": false";
    EXPECT_NE(probed->find(expectedKey, blockPos), std::string::npos);
    EXPECT_EQ(*maxCall > 0.0, expectedOn)
        << "with no key present the lane must follow the probe's verdict on this machine";

    // The control: the same fixture with the forcing key. The flag flips and
    // the delivered bound is the lane's on any machine - while the machine's
    // own reading keeps being reported, unchanged in kind.
    auto forced =
        RunInputText(CertifiedBoundAlkaneToml(*molecule, "force_certified_lane = true\n").c_str());
    ASSERT_TRUE(forced.has_value()) << forced.error().message;
    const auto forcedBlock = forced->find("\"compute_profile\"");
    ASSERT_NE(forcedBlock, std::string::npos);
    EXPECT_NE(forced->find("\"certified_lane_forced\": true", forcedBlock), std::string::npos);
    const auto forcedRatio = JsonNumber(*forced, "\"ratio\":", forcedBlock);
    ASSERT_TRUE(forcedRatio.has_value());
    EXPECT_GT(*forcedRatio, 0.0) << "the forced run still reports the machine it ran on";

    const auto forcedMax = JsonNumber(*forced, "\"max_call_ha\":");
    ASSERT_TRUE(forcedMax.has_value());
    EXPECT_GT(*forcedMax, 0.0);
}

// The same key closes the OTHER half of the compute-profile counterfactual. The
// amendment there records that the same binary with the probe return forced to
// the T1000 reading passes all five CertifiedBound pins - and the enforcement
// pin among them only because its ON arm (the comparison that FALLS BACK)
// needs a live lane. That arm was reachable on this box by no supported path
// either: `enforce_certified_bound` on its own runs vacuously where the lane is
// off, which is the disclosure rule-legal and says nothing about the fall-back the budget is
// there to force.
//
// FALSIFIABILITY. The lane-off reading of this same fixture is pinned by
// CertifiedBoundEnforcementDecidesTheRouting (enforced true with every
// measured member at its observed zero and fell_back_to_fp64 false), so a
// driver that dropped either key fails one of the two tests rather than
// passing both.
TEST(DriverPinTest, ForcedCertifiedLaneMakesTheBudgetFallbackReachable) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto molecule = qcx::testing::MakeAlkaneSto3g(4);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto result = RunInputText(CertifiedBoundEnforcementForcedAlkaneToml(*molecule).c_str());
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(result->find("\"enforced\": true"), std::string::npos);
    EXPECT_NE(result->find("\"fell_back_to_fp64\": true"), std::string::npos);

    // The comparison the fall-back took on: the lane routed quartets, their
    // sum did not fit the preset budget, and the delivered bound is the zero
    // that moved.
    const auto budget = JsonNumber(*result, "\"budget_ha\":");
    const auto routed = JsonNumber(*result, "\"routed_ha\":");
    const auto routedQuartets = JsonNumber(*result, "\"routed_quartets\":");
    ASSERT_TRUE(budget.has_value());
    ASSERT_TRUE(routed.has_value());
    ASSERT_TRUE(routedQuartets.has_value());
    EXPECT_GT(*budget, 0.0);
    EXPECT_GT(*routed, *budget);
    EXPECT_GT(*routedQuartets, 0.0);
    EXPECT_NE(result->find("\"last_call_ha\": 0.0"), std::string::npos);
}

// The O2 triplet's SAD-UHF energy as the machinery (explicit
// `fock_builder = "direct"`) path delivers it: the pyscf-verified dense
// value, held at 5e-7 by O2UhfSadDirectReproducesThePins because the
// threaded contraction picks a fixed point out of the near-degenerate O2
// UHF cluster . Single-sourced so the pins that read it cannot drift
// apart from one another.
constexpr double kO2UhfMachineryEnergy = -147.63394678545018;

// The energy the lean arm delivered BEFORE the unrestricted seam engaged:
// measured bit-identically over four consecutive runs (2026-09-11), the
// lean accumulation being window-order deterministic, and within 8.3e-11
// of the single-threaded dense reference -147.63394656433729.
// SUPERSEDED as a pin on 2026-09-13: with the seam engaged the same
// fixture lands at -147.6339468200857, which is 3.46e-8 from
// kO2UhfMachineryEnergy against this value's 2.21e-7. It is kept not as a
// pin but as the BAR the engaged lean is measured against -
// O2UhfAutoLeanWritesTheLeanMember asserts that ordering, and this
// constant is the disengaged branch's side of it.
constexpr double kO2UhfLeanPreSeamEnergy = -147.63394656425416;

TEST(DriverPinTest, O2UhfSadDirectReproducesThePins) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kO2Toml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // The direct UHF O2 energy pin is 5e-7, not 1e-7: the threaded direct
    // Fock contraction (fock_build.cpp OpenMP/Eigen reductions) perturbs
    // the Fock matrix in its last bits, and DIIS on the near-degenerate
    // UHF landscape amplifies that into a ~2.5e-7 cluster of fixed
    // points that the thread schedule picks from run to run (observed
    // 2026-08-25: the dense reference -147.63394678545018 and the
    // single-threaded -147.63394656433729 both live in it). The pin is
    // the pyscf-verified dense value; 5e-7 covers the cluster without
    // masking real regressions (the H2/UHF suite's other pins stay tight).
    const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, kO2UhfMachineryEnergy, 5e-7);

    const auto spinSquared = JsonNumber(*result, "\"spin_squared\"");
    ASSERT_TRUE(spinSquared.has_value());
    EXPECT_NEAR(*spinSquared, 2.0034108576810308, 1e-6);
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

    // through the driver path: each oxygen carries one unpaired
    // electron (populations_test O2TripletPins), and the neutral O2 dipole
    // is zero by symmetry (multipoles_test O2TripletDipoleAndQuadrupole).
    const auto populationsPos = result->find("\"populations\"");
    ASSERT_NE(populationsPos, std::string::npos);
    const auto mullikenPos = result->find("\"mulliken\"", populationsPos);
    ASSERT_NE(mullikenPos, std::string::npos);
    const auto mullikenSpin = JsonNumberArray(*result, "\"spin\":", mullikenPos);
    ASSERT_TRUE(mullikenSpin.has_value());
    ASSERT_EQ(mullikenSpin->values.size(), 2u);
    EXPECT_NEAR(mullikenSpin->values[0], 1.0, 1e-7);
    EXPECT_NEAR(mullikenSpin->values[1], 1.0, 1e-7);
    const auto dipole = JsonNumberArray(*result, "\"dipole\":");
    ASSERT_TRUE(dipole.has_value());
    ASSERT_EQ(dipole->values.size(), 3u);
    EXPECT_NEAR(dipole->values[0], 0.0, 1e-7);
    EXPECT_NEAR(dipole->values[1], 0.0, 1e-7);
    EXPECT_NEAR(dipole->values[2], 0.0, 1e-7);

    // The explicit family word keeps the machinery member (Ruling B): the
    // Unrestricted UHF seam is reached by the lean selection alone, so the record
    // of a UHF run that named "direct" must still say so.
    EXPECT_NE(result->find("\"builder\": \"direct\""), std::string::npos);
    EXPECT_NE(result->find("\"builder_member\": \"direct\""), std::string::npos);
}

// The OH doublet of STO-3G: R = 1.8326 Bohr on the z axis (0.9697701567008378
// A after the exact conversion), the fixture that gives the unrestricted
// driver path an external reference. It is chosen against O2 above on
// purpose: OH has ONE SCF solution, so a comparison against a foreign code
// needs no fixed-point band and no near-degenerate-cluster commentary -
// whatever the two disagree about is a real disagreement.
//
// The [resources] thread_cap = 1 is a FIXTURE requirement, not a speed knob.
// The CPU parallel primitive runs schedule(dynamic, 1)
// (backend/include/qcx/backend/cpu_backend.hpp:115), so the iteration-to-
// thread assignment - and with it the accumulation order and the SCF's
// low-order bits - varies run to run above a team of one. This fixture used to
// carry a 1e-10 gate, and the serial path converges at 9.756e-11, 2.4% under
// it, so a reordered accumulation flipped the record's "converged" flag:
// measured 2026-09-16 at this machine's resolved 5-thread team, 6 of 10 runs
// reported it, against 20 of 20 bit-identical at one thread, and the aarch64 CI
// leg never reported it at all. The cell below therefore carries the driver's
// default pair (energy 1e-8 / density 1e-6) and a spin_squared band (1e-4)
// measured at it: at 1e-8/1e-6 the diagnostic lands 1.3-2.6e-5 from 0.753257,
// so the gate and the band are one change and neither moves alone.
// One thread keeps the run reproducible, which the band relies on.
const char* kOhToml = R"(
[molecule]
charge = 0
multiplicity = 2
atoms = [["O", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.9697701567008378]]

[basis]
orbital = "sto-3g"

[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kTight"

[scf]
use_diis = true
energy_tolerance = 1e-8
density_tolerance = 1e-6

[guess]
type = "sad"

[resources]
thread_cap = 1
)";
// The gate is the driver's own default pair - energy 1e-8, density 1e-6
// (main.cpp:101-102) - not the 1e-10/1e-08 this cell used to declare. Measured
// at 1e-10: the serial path converges at 9.756e-11, 2.4% under its own gate, so
// a host-varying accumulation order flips the record's "converged" flag (6 of
// 10 runs at a resolved five-thread team, against 20 of 20 bit-identical at one
// thread), and on aarch64 the run does not reach 1e-10 at all. The ENERGY VALUE
// pin (1e-7) and the ROHF bound below are unchanged by the move; the
// spin_squared diagnostic is not, and it is the check that pays for it: at this
// looser gate the density fixed point is a different one, so the diagnostic
// moves by ~1e-5 and the band below is measured at the new gate and not
// inherited from the 1e-10 one. thread_cap = 1 keeps the serial path, so the
// value the band is measured against does not move with the team size.

// The external references (pyscf 2.14.0, gto.M with the coordinates in
// Angstrom - the file's unit, run_input.hpp:5 - conv_tol 1e-12, minao guess),
// each at the Bohr bond length its own fixture converts to:
//   OH, R = 1.8326 Bohr: UHF -74.36264494805 (<S^2> 0.753257),
//                        ROHF -74.36153805173 (<S^2> 0.750000).
//   O2, R = 2.2818443 Bohr: ROHF -147.63216699080 (<S^2> 2.000000).
// The ROHF values are the variational BAR, not a second energy pin: a
// spin-restricted determinant is a determinant with the same nAlpha/nBeta,
// so it lies inside the unrestricted variational space and the unrestricted
// MINIMUM can never sit above it. A converged unrestricted run above the
// ROHF bound therefore did not find the unrestricted minimum, whatever its
// gate reports - the invariant a mis-set occupation or a broken J/K
// assembly violates loudly and that no energy pin can see, since an energy
// pin only says "not the number I was shown before".
constexpr double kOhUhfReferenceEnergy = -74.36264494805;
constexpr double kOhRohfBoundEnergy = -74.36153805173;
constexpr double kO2RohfBoundEnergy = -147.63216699080;

// The driver-side unrestricted rows against a code that is not qcx, and the
// variational bound on both fixtures. Measured at landing, OH pinned recipe:
// -74.36264497493 / -74.36264497493 / -74.36264497472 over three runs
// (spread 2.1e-10, agreement 2.7e-8), so 1e-7 is the band and the spread is
// the thread/DIIS cluster, not a drift.
TEST(DriverPinTest, OhUhfMatchesTheExternalReferenceAndStaysUnderTheRohfBound) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto oh = RunInputText(kOhToml);
    ASSERT_TRUE(oh.has_value()) << oh.error().message;
    const auto ohTotal = JsonNumber(*oh, "\"total_energy_hartree\"");
    ASSERT_TRUE(ohTotal.has_value());
    EXPECT_NEAR(*ohTotal, kOhUhfReferenceEnergy, 1e-7);
    EXPECT_LT(*ohTotal, kOhRohfBoundEnergy);
    const auto ohSpin = JsonNumber(*oh, "\"spin_squared\"");
    ASSERT_TRUE(ohSpin.has_value());
    // The band is MEASURED at this cell's own gate, the driver's default pair
    // (energy 1e-8 / density 1e-6): MSVC Release on this tree, 2026-09-20, the
    // run reports 0.75323082449958978 against the external reference 0.753257 -
    // 2.6176e-5 away, bit-identical over four runs, which is what thread_cap = 1
    // buys. This cell's header records a 1.3e-5 to 2.6e-5 spread from the same
    // pin at the same pair, and that spread is what the band has to cover, so
    // 1e-4 is the band: 3.8x this reading and 7.7x the low end of it. It stays
    // a real check at that width - a collapsed, spin-restricted solution reads
    // the ROHF value pinned above, 0.750000, which is 32x the band away.
    EXPECT_NEAR(*ohSpin, 0.753257, 1e-4);
    EXPECT_NE(oh->find("\"converged\": true"), std::string::npos);

    // The same bound on the near-degenerate fixture, so the invariant is not
    // sheltered by the fact that OH is well behaved. The SAD start is the
    // recipe the O2 pins above already hold; the bound is checked there, at
    // the geometry this file's own Å conversion produces.
    auto o2 = RunInputText(kO2Toml);
    ASSERT_TRUE(o2.has_value()) << o2.error().message;
    const auto o2Total = JsonNumber(*o2, "\"total_energy_hartree\"");
    ASSERT_TRUE(o2Total.has_value());
    EXPECT_LT(*o2Total, kO2RohfBoundEnergy);
}

// The unrestricted seam, end to end and through the run record: the O2 triplet
// with the [method].fock_builder key ABSENT. At nBasis 10 <= the 1000-basis-
// function ceiling the UHF path now wires the direct family's within-family
// lean member (before the seam the same fixture ran the machinery), so the
// selection record must say so - the same member name
// H2AutoLeanWritesTheLeanMember pins on the RHF side, on the leg that had
// no lean arm at all.
TEST(DriverPinTest, O2UhfAutoLeanWritesTheLeanMember) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kO2AutoLeanToml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(result->find("\"builder\": \"direct\""), std::string::npos);
    EXPECT_NE(result->find("\"builder_member\": \"lean\""), std::string::npos);

    // The lean member has no workspace budget and no Create-time mode
    // record to report, so the record's blocks stay ABSENT rather than
    // arriving as zero-filled stand-ins (the null honesty policy - the RHF
    // lean leg's shape).
    EXPECT_EQ(result->find("\"workspace_budget\""), std::string::npos);
    EXPECT_EQ(result->find("\"mode_record\""), std::string::npos);
    EXPECT_EQ(result->find("\"exchange_mode_record\""), std::string::npos);

    // The lean UHF leg converges on the O2 triplet. The assertion is the
    // ORDERING, not an absolute band (owner ruling, 2026-09-13): the
    // engaged lean must land closer to the machinery pin than the
    // disengaged lean did. The 1e-8 band this test carried was never sound
    // on this fixture - the O2 triplet is a near-degenerate UHF fixed-point
    // cluster whose two pins already sit 2.21e-7 apart, 22x the band -
    // and on 2026-09-13 it failed with a delta of 2.5583153728803154e-07 on
    // a value that is 3.46e-8 from the machinery pin, 6.4x CLOSER to it
    // than the branch it was being held against. Re-pinning to the engaged
    // number or widening the band would each delete that signal; the
    // ordering keeps it. The bar is numerical quality only: that the seam
    // ENGAGED is what the record assertions above hold, not this.
    const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    const double engagedDistance = std::abs(*total - kO2UhfMachineryEnergy);
    const double disengagedDistance = std::abs(kO2UhfLeanPreSeamEnergy - kO2UhfMachineryEnergy);
    EXPECT_LT(engagedDistance, disengagedDistance)
        << "the engaged lean sits " << engagedDistance << " from the machinery pin "
        << kO2UhfMachineryEnergy << "; the disengaged lean sat " << disengagedDistance
        << " away - the lean seam must not land the answer further from the reference "
           "than the branch it replaced";

    const auto spinSquared = JsonNumber(*result, "\"spin_squared\"");
    ASSERT_TRUE(spinSquared.has_value());
    EXPECT_NEAR(*spinSquared, 2.003410846644341, 1e-6);
}

// The explicit `fock_builder = "lean"` spelling on the UHF leg: one
// within-family request, the RHF word's twin (no family word, so the
// record's family stays "direct" and the lean flag carries the choice).
// Both spellings must reach the SAME arm and produce the SAME run - a
// second spelling that took a different path would be the silent
// substitution the word's own reasoning text exists to prevent - so the
// two fixtures are run here and their energies compared EXACTLY.
TEST(DriverPinTest, O2UhfExplicitLeanWordWiresTheLeanMember) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kO2LeanWordToml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(result->find("\"builder\": \"direct\""), std::string::npos);
    EXPECT_NE(result->find("\"builder_member\": \"lean\""), std::string::npos);

    // The lean word names no family word, so the record's explicit_builder
    // stays absent (the null honesty policy) and the reasoning is the
    // record of the request.
    EXPECT_NE(result->find("explicit fock_builder lean honored"), std::string::npos);
    EXPECT_EQ(result->find("\"explicit_builder\""), std::string::npos);

    auto autoResult = RunInputText(kO2AutoLeanToml);
    ASSERT_TRUE(autoResult.has_value()) << autoResult.error().message;

    const auto explicitEnergy = JsonNumber(*result, "\"total_energy_hartree\"");
    const auto autoEnergy = JsonNumber(*autoResult, "\"total_energy_hartree\"");
    ASSERT_TRUE(explicitEnergy.has_value());
    ASSERT_TRUE(autoEnergy.has_value());
    EXPECT_DOUBLE_EQ(*explicitEnergy, *autoEnergy);
}

TEST(DriverPinTest, O2UhfComposedQfmmReproducesTheDirectUhfPin) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kO2QfmmToml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // The composed-UHF gate: the O2 run through the composed QFMM
    // UHF path (the explicit qfmm UHF branch of RunDriver) must reproduce
    // the direct path's pinned ground state. At kTight on sto-3g the
    // QFMM half's error sits far inside the 5e-7 band that already covers
    // the direct run's DIIS/thread fixed-point cluster (the direct pin
    // test's comment above), and the SAD-seeded plain iteration follows
    // the same trajectory to the same fixed point - a composition bug (a
    // dropped spin channel, the double-H trap, a wrong halving of the
    // total density) would land at chemical scale instead.
    const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, kO2UhfMachineryEnergy, 5e-7);

    // The composed-UHF model disclosure: the UHF runner carries
    // the engine's own record out of RunQfmmUhfScf, so an unrestricted QFMM
    // run answers the same question the RHF one does (the DriverRunTest
    // disclosure test holds the RHF legs). The kTight rung's gate is visible
    // here as the resolved theta 0.0 with the far field it admits: under the
    // centre-to-width test theta <= 0 is "never well separated", so at kTight
    // the absent-theta default keeps the exact near-field-only rung rather
    // than silently acquiring a live far field - and the record states that as
    // the run's own geometric far-pair count, which is the fact the rung is
    // read against rather than the name of the test that decided it.
    EXPECT_NE(result->find("\"qfmm_model\""), std::string::npos);
    EXPECT_NE(result->find("\"extent_model\": \"kProductBall\""), std::string::npos);
    EXPECT_NE(result->find("\"theta\": 0.0"), std::string::npos);
    EXPECT_NE(result->find("\"geometric_far_pair_count\": 0,"), std::string::npos);

    const auto spinSquared = JsonNumber(*result, "\"spin_squared\"");
    ASSERT_TRUE(spinSquared.has_value());
    EXPECT_NEAR(*spinSquared, 2.0034108576810308, 1e-6);
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

    // through the driver path, asserted like the direct pin
    // test: each oxygen carries one unpaired electron and the neutral O2
    // dipole is zero by symmetry.
    const auto populationsPos = result->find("\"populations\"");
    ASSERT_NE(populationsPos, std::string::npos);
    const auto mullikenPos = result->find("\"mulliken\"", populationsPos);
    ASSERT_NE(mullikenPos, std::string::npos);
    const auto mullikenSpin = JsonNumberArray(*result, "\"spin\":", mullikenPos);
    ASSERT_TRUE(mullikenSpin.has_value());
    ASSERT_EQ(mullikenSpin->values.size(), 2u);
    EXPECT_NEAR(mullikenSpin->values[0], 1.0, 1e-7);
    EXPECT_NEAR(mullikenSpin->values[1], 1.0, 1e-7);
    const auto dipole = JsonNumberArray(*result, "\"dipole\":");
    ASSERT_TRUE(dipole.has_value());
    ASSERT_EQ(dipole->values.size(), 3u);
    EXPECT_NEAR(dipole->values[0], 0.0, 1e-7);
    EXPECT_NEAR(dipole->values[1], 0.0, 1e-7);
    EXPECT_NEAR(dipole->values[2], 0.0, 1e-7);

    // The budget/mode records (the composed-UHF records gap fix,
    // interrogation 6): the composed run now joins the budget path. The
    // default 16.0 GiB cap becomes a workspace budget of the cap MINUS the
    // run's reserve, which is the number the pre-gate setup arm refuses on
    // handed back so that the arm's floor and the engine's rung condition
    // are one quantity (integrals::SetupAdmissionReserveBytes = the engine's
    // own ramp-and-sweeps bound SetupRampPeakBytes times its one slack
    // constant kSetupAdmissionSlack, 1.5). At this fixture - O2/sto-3g, no
    // aux in effect - the bound is 320,280 B, so the reserve is the
    // 480,420 B above it and the capacity is 16.0 * 2^30 - 480,420 =
    // 17179388764 bytes. (The deleted model's fitted base used to be
    // subtracted here and the pre-deletion pin recorded 17119739641;
    // nothing of that model is restored.) The commit covers
    // BOTH nested Creates (the QFMM half's estimate
    // reserves first, the exchange half's second - nesting order =
    // reservation order).
    EXPECT_NE(result->find("\"workspace_budget\""), std::string::npos);
    EXPECT_NE(result->find("\"capacity_bytes\": 17179388764"), std::string::npos);
    EXPECT_NE(result->find("\"committed_bytes\""), std::string::npos);

    // The mode records: mode_record carries the Coulomb (QFMM) half's
    // Create-time decision (kFastPath at this tiny system, outer store
    // term live), exchange_mode_record the exchange-only half's - both
    // admission-gate observations must be visible, the direct-UHF
    // surface's shape.
    const auto modeRecordPos = result->find("\"mode_record\"");
    ASSERT_NE(modeRecordPos, std::string::npos);
    EXPECT_NE(result->find("\"mode\": \"kFastPath\""), std::string::npos);
    EXPECT_NE(result->find("\"exchange_mode_record\""), std::string::npos);

    // Both records carry an "outer_store_bytes" member, and nlohmann
    // serializes object keys lexicographically, so exchange_mode_record
    // precedes mode_record and a first-occurrence read would serve the
    // exchange-only half's 0. Scope the read to mode_record (its members
    // dump contiguously): the Coulomb half's QFMM outer store is live.
    const auto outerStore = JsonNumber(*result, "\"outer_store_bytes\"", modeRecordPos);
    ASSERT_TRUE(outerStore.has_value());
    EXPECT_GT(*outerStore, 0.0);
}

// The unrestricted RI-J-link run path end to end (the per-spin adapter
// landed here, MakeRiJLinkUhfFockBuilder over RiJkFockBuilder's two exposed
// halves), on the OPEN-shell O2 recipe: the SAD seed, the tight gate and the
// per-spin DIIS are kO2Toml's, so what this row changes is the Fock assembly
// alone - and the spin structure it reads comes from a genuinely polarized
// (alpha != beta) density pair.
//
// FALSIFIABILITY, and the accept-and-drop mutation this row is built to
// catch: a wiring that RESOLVED ri_j_link but dispatched the direct family
// (the gpu_split defect class) would still publish "builder": "ri_j_link" -
// the selection is echoed from the request - so the name is not the
// evidence. The evidence is the RI engine's own Create-time record:
// mode_record's RI-family byte terms (tensor_bytes, ri_matrix_bytes,
// metric_bytes) are written by RiJkFockBuilder's Create and are ZERO on
// every direct-family record (FockBuildOptions fills none of them), so
// tensor_bytes > 0 is a measurement of which engine built this run's Fock.
// Before this wiring landed the request never reached a run at all (the UHF
// whitelist refused ri_j_link by name), which is why this test's
// predecessor was a refusal cell.
TEST(DriverPinTest, O2UhfRiJLinkWiresThePerSpinAdapterEndToEnd) {
    if (qcx::testing::IsFastOnlyMode() || !qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "heavy numerical test: Release-only; the universal-J aux f shells "
                        "exceed this build's kMaxEngineL (CI lmax=2)";
    }

    auto result = RunInputText(kO2RiJLinkToml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(result->find("\"builder\": \"ri_j_link\""), std::string::npos);
    EXPECT_NE(result->find("\"builder_member\": \"ri_j_link\""), std::string::npos);

    // The engine's own record, scoped past the first occurrence (the JSON
    // serializes object keys lexicographically and the mode block's members
    // dump contiguously - the QFMM pin's own reading rule next door).
    const auto modeRecordPos = result->find("\"mode_record\"");
    ASSERT_NE(modeRecordPos, std::string::npos) << *result;
    const auto tensorBytes = JsonNumber(*result, "\"tensor_bytes\"", modeRecordPos);
    ASSERT_TRUE(tensorBytes.has_value()) << *result;
    EXPECT_GT(*tensorBytes, 0.0)
        << "the run reported no RI-family tensor bytes: the Fock was not built by the RI engine "
           "(the accept-and-drop wiring this row exists to catch)";

    // The physics, from the open-shell deck: the SAD-seeded plain iteration
    // reaches the same unpolarized-start UHF branch the direct and composed-
    // QFMM pins reach - each oxygen carrying one unpaired electron and the
    // neutral O2 dipole zero by symmetry. MEASURED on this fixture at the
    // time of landing (kTight, sto-3g + the auto-selected def2-universal-jfit,
    // 63 iterations each): the RI-J leg -147.63521409352643, the direct leg
    // -147.63394682039947 (3.5e-8 from the suite's machinery pin), so the
    // energy gap is 1.268e-3 Ha - the RI-J approximation at THIS pairing, and
    // why the band below is 5e-3 rather than the 5e-7 the composed-QFMM pin
    // uses: sto-3g x universal-jfit is the aux rule's weakest pairing
    // (aux_basis.hpp: 6.99x the control), and the two legs' spin structure
    // agrees far more tightly than their energies - spin-squared 2.0034097 vs
    // 2.0034109 (1.2e-6 apart), the per-atom spin 1.0 both ways.
    //
    // What the band can and cannot establish: it catches a chemistry-scale
    // Coulomb defect on the open-shell branch (a wrong J scaling, a spin-
    // summed density reaching an exchange half, a dropped channel), because
    // those move the energy orders of magnitude further than the aux
    // approximation does. It says NOTHING about the RI-J fit quality - that
    // is the engine's own acceptance (ri_engine_test.cpp) - and it is not a
    // tolerance argument for the adapter: the identity that establishes the
    // adapter's algebra exactly is
    // UhfRiJLinkMatchesTheRestrictedLegOnAClosedShellDeck, where the shared
    // engine cancels out of the comparison.
    const auto energy = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(energy.has_value());
    EXPECT_NEAR(*energy, kO2UhfMachineryEnergy, 5e-3)
        << "the unrestricted RI-J-link run did not land on the machinery pin's branch";

    const auto spinSquared = JsonNumber(*result, "\"spin_squared\"");
    ASSERT_TRUE(spinSquared.has_value());
    EXPECT_NEAR(*spinSquared, 2.0034108576810308, 1e-4);

    const auto populationsPos = result->find("\"populations\"");
    ASSERT_NE(populationsPos, std::string::npos) << *result;
    const auto mullikenPos = result->find("\"mulliken\"", populationsPos);
    ASSERT_NE(mullikenPos, std::string::npos) << *result;
    const auto mullikenSpin = JsonNumberArray(*result, "\"spin\":", mullikenPos);
    ASSERT_TRUE(mullikenSpin.has_value()) << *result;
    ASSERT_EQ(mullikenSpin->values.size(), 2u);
    EXPECT_NEAR(mullikenSpin->values[0], 1.0, 1e-5);
    EXPECT_NEAR(mullikenSpin->values[1], 1.0, 1e-5);

    const auto dipole = JsonNumberArray(*result, "\"dipole\":");
    ASSERT_TRUE(dipole.has_value()) << *result;
    ASSERT_EQ(dipole->values.size(), 3u);
    EXPECT_NEAR(dipole->values[0], 0.0, 1e-6);
    EXPECT_NEAR(dipole->values[1], 0.0, 1e-6);
    EXPECT_NEAR(dipole->values[2], 0.0, 1e-6);

    // The certified bound takes the same shorter BuildFock path it takes on
    // the restricted ri_j_link leg (the seam hands it the two-argument form,
    // so no bound pointer reaches the fp32 lane its nested exchange half
    // runs): the key must be ABSENT, never a fabricated zero. The instrument
    // check is the sibling pin CertifiedBoundIsAbsentOnTheRiJLink, which
    // finds the key on the SAME fixture's machinery leg.
    EXPECT_EQ(result->find("certified_bound"), std::string::npos) << *result;

    // The memory audit block records the RI family's own (n, nAux) terms -
    // the same ones the restricted leg records: this leg's cap was weighed
    // against the RI-J model, not the direct family's.
}

// The per-spin adapter's correctness pin, and the deck is chosen so that the
// comparison is an IDENTITY rather than a tolerance: H2O/def2-SVP at
// multiplicity 1, so the unrestricted run's two spin densities are equal
// (P_alpha = P_beta = D/2).
//
// Why the two legs MUST agree there, exactly: the restricted leg builds
// F = H + 2 J_RI(rho) - K(rho) with rho = D/2; the unrestricted adapter
// builds F_sigma = 2 J_RI(0.5 (P_alpha + P_beta)) + (H - K(P_sigma)), which
// at P_alpha = P_beta = D/2 is 2 J_RI(D/2) - K(D/2) + H - the SAME matrix,
// element for element, because J_RI and K are both linear in the density
// they are handed and 0.5 (P_alpha + P_beta) = D/2. A defect in the
// adapter's two density conventions (a missing halving, a doubled J, the
// spin-summed density fed to an exchange half), in the H accounting (the
// subtraction the DIRECT family's split halves need and this family's do
// not), or in the per-spin assignment would put a chemical-scale difference
// between these two numbers - while the symmetric deck keeps the identity
// free of any approximation argument: the two runs use ONE RI-J engine, so
// its own error cancels out of the comparison rather than entering it.
//
// What the identity does NOT establish, stated because the number could be
// read as proving more: the two runs reach their fixed points through
// different SCF loops (RunRhfScf and RunUhfScf), so their energies agree to
// the loops' convergence gates, not to the last bit; and an alpha/beta
// exchange would be invisible on a deck where alpha == beta. The open-shell
// row above covers the polarized pair, and the split halves' algebra
// (BuildCoulombOnly + BuildExchangeOnly composing to BuildFock) is pinned
// at the engine's own level. A tautology this is not: the RHF leg runs the
// FUSED BuildFock, the UHF leg runs three split calls through a different
// runner, dispatch branch, seam and loop, and only the physics makes the
// two the same matrix.
TEST(DriverPinTest, UhfRiJLinkMatchesTheRestrictedLegOnAClosedShellDeck) {
    if (qcx::testing::IsFastOnlyMode() || !qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "heavy numerical test: Release-only; the universal-J aux f shells "
                        "exceed this build's kMaxEngineL (CI lmax=2)";
    }

    auto restricted = RunInputText(H2oDef2SvpToml("ri_j_link").c_str());
    ASSERT_TRUE(restricted.has_value()) << restricted.error().message;
    EXPECT_NE(restricted->find("\"converged\": true"), std::string::npos);

    auto unrestricted = RunInputText(H2oDef2SvpUhfToml("ri_j_link").c_str());
    ASSERT_TRUE(unrestricted.has_value()) << unrestricted.error().message;
    EXPECT_NE(unrestricted->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(unrestricted->find("\"builder\": \"ri_j_link\""), std::string::npos);

    const auto restrictedEnergy = JsonNumber(*restricted, "\"total_energy_hartree\"");
    const auto unrestrictedEnergy = JsonNumber(*unrestricted, "\"total_energy_hartree\"");
    ASSERT_TRUE(restrictedEnergy.has_value());
    ASSERT_TRUE(unrestrictedEnergy.has_value());
    // MEASURED on this fixture at the time of landing (kLoose, def2-SVP +
    // def2-universal-jfit, 12 iterations each): -75.96112397489613 (RHF) and
    // -75.9611239748961 (UHF) - 2.8e-14 apart, four orders inside the 1e-8
    // band below, which is the runs' own energy gate: the two loops reach one
    // fixed point, so what separates them is convergence bookkeeping rather
    // than physics. A tolerance pinned at the gate rather than at the
    // measurement is deliberate - the measured value would flake on a
    // different thread count, and a defect here is chemistry-scale.
    EXPECT_NEAR(*unrestrictedEnergy, *restrictedEnergy, 1e-8)
        << "restricted " << *restrictedEnergy << ", unrestricted " << *unrestrictedEnergy;
}

// The unrestricted composed-full-RI run path end to end (the per-spin adapter
// landed here, MakeRiFullUhfFockBuilder over
// RiFullFockBuilder::BuildFockHalves): the OPEN-shell O2 recipe, so the spin
// structure it reads comes from a genuinely polarized (alpha != beta) density
// pair - and a spurious coupling between the two channels would move the
// per-atom spins, which the closed-shell identity pin below cannot see.
//
// FALSIFIABILITY, and the accept-and-drop mutation this row is built to
// catch: a wiring that RESOLVED ri_jk but dispatched another family would
// still publish "builder": "ri_jk" (the selection is echoed from the request),
// still name its member, and still emit the approximation block (that record
// is built from the RESOLVED kind, not from the run) - so none of those is
// evidence. The evidence is the composed builder's OWN Create-time record,
// which only this family can produce: resources_resolved.ri_jk_mode is filled
// from RiFullFockBuilder::ModeInfo() by this arm alone, and its tensor_bytes
// is the engine's own term decomposition - a number no direct-family run has
// anywhere to write. Before this wiring landed the request never reached a run at all
// (the unrestricted whitelist refused ri_jk by name), which is why this test's
// predecessor was a refusal arm of DriverErrorTest.
TEST(DriverPinTest, UhfRiJkWiresThePerSpinAdapterEndToEnd) {
    if (qcx::testing::IsFastOnlyMode() || !qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "heavy numerical test: Release-only; the JK-fit aux f shells exceed "
                        "this build's kMaxEngineL (CI lmax=2)";
    }

    auto result = RunInputText(kO2RiJkToml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_NE(result->find("\"converged\": true"), std::string::npos) << *result;
    EXPECT_NE(result->find("\"builder\": \"ri_jk\""), std::string::npos);
    EXPECT_NE(result->find("\"builder_member\": \"occ_ri_k\""), std::string::npos);
    EXPECT_NE(result->find("\"explicit_builder\": \"ri_jk\""), std::string::npos);

    // The engine's own record, scoped past the mode block's own opening (the
    // JSON serializes object keys lexicographically and the block's members
    // dump contiguously - the sibling pins' reading rule).
    const auto modePos = result->find("\"ri_jk_mode\"");
    ASSERT_NE(modePos, std::string::npos)
        << "the run carries no ri_jk_mode block, so no composed full-RI builder was created: "
           "this is the accept-and-drop wiring this row exists to catch: "
        << *result;
    const auto tensorBytes = JsonNumber(*result, "\"tensor_bytes\"", modePos);
    ASSERT_TRUE(tensorBytes.has_value()) << *result;
    EXPECT_GT(*tensorBytes, 0.0)
        << "the composed builder reported no tensor bytes: the Fock was not built by it";
    EXPECT_NE(result->find("\"rung\": \"kFast\""), std::string::npos) << *result;

    // The two inherited disclosures, both read here because both are the ones
    // a reader of an unrestricted ri_jk energy needs and neither is re-decided
    // by this leg. The approximation block is built in RunDriver from the
    // RESOLVED kind, never from the SCF leg - so this run owes it
    // exactly as its restricted twin does, and the aux it names is the JK
    // KIND's own resolution (def2-svp auto-selects def2-universal-jkfit for
    // ri_jk and def2-universal-jfit for ri_j_link). The measured-error
    // disclosure rides the same block - and its fixture is a
    // CLOSED-SHELL water cell, which is the open item this wiring records rather
    // than fixes: an unrestricted run discloses the restricted family's worst
    // characterized cell.
    EXPECT_NE(result->find("\"approximation\""), std::string::npos) << *result;
    EXPECT_NE(result->find("\"exchange\": \"occ_ri_k\""), std::string::npos) << *result;
    EXPECT_NE(result->find("\"aux_basis\": \"def2-universal-jkfit\""), std::string::npos)
        << *result;
    EXPECT_NE(result->find("\"exchange_error\""), std::string::npos) << *result;
    EXPECT_NE(result->find("\"measured_on\": \"h2o_sto3g\""), std::string::npos) << *result;

    // The physics, from the open-shell deck: the SAD-seeded plain iteration
    // reaches the same unpolarized-start UHF branch the direct, composed-QFMM
    // and ri_j_link pins reach - each oxygen carrying one unpaired electron
    // and the neutral O2 dipole zero by symmetry.
    //
    // What the band can and cannot establish: it catches a chemistry-scale
    // defect on the open-shell branch (a wrong J scaling, a spin-summed
    // density reaching an exchange half, a dropped channel, a doubled H),
    // because those move the energy far further than the RI-K fit does. It
    // says NOTHING about the fit's quality - that is the engine's own
    // acceptance - and it is NOT the adapter's correctness argument: the
    // identity that establishes the assembly's algebra exactly is
    // UhfRiJkMatchesTheRestrictedLegOnAClosedShellDeck, where one engine
    // cancels out of the comparison. The band is the sibling ri_j_link pin's
    // 5e-3, deliberately: the same branch, the same aux rule's weakest
    // pairing (sto-3g x a universal fit), and a defect of the class this row
    // hunts is orders of magnitude larger than either family's fit error.
    const auto energy = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(energy.has_value());
    const auto iterations = JsonNumber(*result, "\"iterations\"");
    ASSERT_TRUE(iterations.has_value());
    std::printf("UHF ri_jk O2/STO-3G (KTight, def2-universal-jkfit): total = %.15e in %.0f "
                "iterations\n",
                *energy,
                *iterations);
    EXPECT_NEAR(*energy, kO2UhfMachineryEnergy, 5e-3)
        << "the unrestricted ri_jk run did not land on the machinery pin's branch";

    const auto spinSquared = JsonNumber(*result, "\"spin_squared\"");
    ASSERT_TRUE(spinSquared.has_value());
    EXPECT_NEAR(*spinSquared, 2.0034108576810308, 1e-3);

    const auto populationsPos = result->find("\"populations\"");
    ASSERT_NE(populationsPos, std::string::npos) << *result;
    const auto mullikenPos = result->find("\"mulliken\"", populationsPos);
    ASSERT_NE(mullikenPos, std::string::npos) << *result;
    const auto mullikenSpin = JsonNumberArray(*result, "\"spin\":", mullikenPos);
    ASSERT_TRUE(mullikenSpin.has_value()) << *result;
    ASSERT_EQ(mullikenSpin->values.size(), 2u);
    // The per-atom spin is 1 to the FITTED EXCHANGE's footprint, not to the
    // arithmetic's: MEASURED on this fixture at the time of landing,
    // 1.0025983433475441 / 0.99740165665245328 - 2.6e-3 from unity, where the
    // ri_j_link sibling on the same deck holds 1e-5. That gap is the operator
    // being approximated and not a difference between two implementations:
    // ri_j_link fits the COULOMB half, which is spin-independent and barely
    // moves the spin structure (its own record's spin-squared sits 1.2e-6 from
    // the direct leg's), while ri_jk fits the EXCHANGE - the operator that
    // sets it. The band is therefore the fit's own scale rather than an
    // exact-exchange family's tolerance, and it still catches what this row is
    // for: a channel swap, a leaked spin-summed density or the other
    // near-degenerate O2 fixed point moves this number by 0.1 to 1, two to
    // twenty times the band.
    //
    // The BAND is 5e-2, set from the fit's measured spread rather than from a
    // round number: the footprint has read 2.6e-3 at landing and 1.265e-2 on
    // the linux-x86 gcc leg (2026-09-19, CI run 35451596080) - a factor of
    // five between environments, because it is the approximation's error and
    // not the arithmetic's - so the band is ten times the energy band this row
    // allows the same fit (5e-3) and twice the worst reading. A round 1e-2 was
    // narrower than the quantity it measures, which pins the machine rather
    // than the code.
    //
    // The two relations below are what the adapter owes EXACTLY, and they
    // carry the row: the pair sums to the run's own total spin whatever the
    // fit does to the split, and the two oxygens are symmetry-equivalent, so
    // the footprint must divide evenly between them.
    EXPECT_NEAR(mullikenSpin->values[0] + mullikenSpin->values[1], 2.0, 1e-9)
        << "the per-spin populations must sum to the run's total spin population";
    EXPECT_NEAR(
        std::abs(mullikenSpin->values[0] - 1.0), std::abs(mullikenSpin->values[1] - 1.0), 1e-12)
        << "the two oxygens are symmetry-equivalent: the fit's footprint must divide evenly";
    EXPECT_NEAR(mullikenSpin->values[0], 1.0, 5e-2);
    EXPECT_NEAR(mullikenSpin->values[1], 1.0, 5e-2);

    // The certified bound takes the shorter BuildFock path it takes on every
    // other RI leg: the pair's seam hands no bound pointer to the fp32 lane,
    // so the key must be ABSENT, never a fabricated zero.
    EXPECT_EQ(result->find("certified_bound"), std::string::npos) << *result;

    // The memory audit block records THIS family's (n, nAux) terms - the same
    // terms the restricted ri_jk leg records, not the direct family's.
}

// The per-spin adapter's correctness pin for the composed full-RI family, on
// the deck where the comparison is an IDENTITY rather than a tolerance:
// H2O/def2-SVP with multiplicity 1, so alpha == beta and the restricted and
// unrestricted legs MUST reach one number.
//
// Why the two legs MUST agree there, exactly: the restricted leg builds
// F = H + 2 J_RI(rho) - K(rho) with rho = D/2 (the fused BuildFock); the
// unrestricted adapter builds F_sigma = H + C(P_sigma) + C(P_other) - K(P_sigma)
// from ONE BuildFockHalves call per spin, which at P_alpha = P_beta = D/2 is
// H + 2 J_RI(D/2) - K_RI(D/2) - the SAME matrix, element for element, because
// both halves are linear in the density they are handed and the per-spin
// argument is the raw density (a halving here would halve J; a doubled J, an
// H added twice, a spin-summed density reaching an exchange half, or the
// direct family's per-channel H subtraction incorrectly applied would each put
// a chemical-scale difference between these two numbers). The symmetric deck
// keeps the identity free of any approximation argument: one engine, so its
// own fit error cancels out of the comparison rather than entering it.
//
// What the identity does NOT establish, stated because the number could be
// read as proving more: the two runs reach their fixed points through
// different SCF loops (RunRhfScf and RunUhfScf), so their energies agree to
// the loops' convergence gates, not necessarily to the last bit; the
// occupied-block factorization this seam derives from the density (the
// restricted arm's own recipe, applied per spin) is exact for a converged
// density and approximate for a DIIS-extrapolated one; and an alpha/beta
// exchange would be invisible on a deck where alpha == beta - the open-shell
// row above covers the polarized pair. A tautology this is not: the two legs
// run different engines' entry points (BuildFock's fused composition against
// two pair calls through a different runner, dispatch branch, seam and loop),
// and only the physics makes the two the same matrix.
TEST(DriverPinTest, UhfRiJkMatchesTheRestrictedLegOnAClosedShellDeck) {
    if (qcx::testing::IsFastOnlyMode() || !qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "heavy numerical test: Release-only; the JK-fit aux f shells exceed "
                        "this build's kMaxEngineL (CI lmax=2)";
    }

    auto restricted = RunInputText(H2oDef2SvpToml("ri_jk").c_str());
    ASSERT_TRUE(restricted.has_value()) << restricted.error().message;
    EXPECT_NE(restricted->find("\"converged\": true"), std::string::npos);

    auto unrestricted = RunInputText(H2oDef2SvpUhfToml("ri_jk").c_str());
    ASSERT_TRUE(unrestricted.has_value()) << unrestricted.error().message;
    EXPECT_NE(unrestricted->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(unrestricted->find("\"builder\": \"ri_jk\""), std::string::npos);
    // The unrestricted leg reaches the composed builder at all - the identity
    // below would be satisfied by two runs of the SAME fused path only if the
    // unrestricted one had dispatched the restricted arm, which is the
    // mutation this assertion and the sibling row's ri_jk_mode measurement
    // together exclude.
    EXPECT_NE(unrestricted->find("\"ri_jk_mode\""), std::string::npos) << *unrestricted;

    const auto restrictedEnergy = JsonNumber(*restricted, "\"total_energy_hartree\"");
    const auto unrestrictedEnergy = JsonNumber(*unrestricted, "\"total_energy_hartree\"");
    ASSERT_TRUE(restrictedEnergy.has_value());
    ASSERT_TRUE(unrestrictedEnergy.has_value());
    std::printf("RHF/UHF ri_jk H2O/def2-SVP closed shell: restricted = %.15e, unrestricted = "
                "%.15e, difference = %.3e\n",
                *restrictedEnergy,
                *unrestrictedEnergy,
                *unrestrictedEnergy - *restrictedEnergy);
    // MEASURED on this fixture at the time of landing - see the printed line
    // above and the record of this wiring for the value and the revision it was taken
    // on. The band is the runs' own energy gate (the defaults rule default, 1e-8), not
    // the measurement: a tolerance pinned at the measurement would flake on a
    // different thread count, and every defect this identity exists to catch
    // is chemistry-scale - four or more orders outside the band.
    EXPECT_NEAR(*unrestrictedEnergy, *restrictedEnergy, 1e-8)
        << "restricted " << *restrictedEnergy << ", unrestricted " << *unrestrictedEnergy;
}

// The one shape of the per-spin adapter no other fixture reaches: a spin
// channel with NO electrons. A doublet whose electron count is odd leaves
// nBeta = 0, and this seam - unlike its siblings - has to DERIVE an occupied
// block from the density it is handed (the scf seam passes densities only),
// so that channel hands it a zero density whose exact factorization is an
// EMPTY matrix - which the builder refuses by name. The floor the seam applies
// instead (one column, driven to zero by the sqrt clamp) is what makes such a
// run an ordinary Hartree-Fock one rather than a refusal, and it adds no
// approximation: with the block exactly zero the exchange half is K(0) = 0 and
// the Coulomb half never reads the block at all.
//
// FALSIFIABILITY: drop the floor and pass the spin's count straight through,
// and this run stops with the builder's own "the occupied orbital block must
// be a non-empty matrix". Every other ri_jk fixture has both channels
// populated, so nothing else in the suite reaches this branch - which is why
// it is a cell rather than a comment.
TEST(DriverPinTest, UhfRiJkRunsASpinChannelWithNoElectrons) {
    if (qcx::testing::IsFastOnlyMode() || !qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "heavy numerical test: Release-only; the JK-fit aux f shells exceed "
                        "this build's kMaxEngineL (CI lmax=2)";
    }

    auto result = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 2
atoms = [["H", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "ri_jk"
accuracy = "kNormal"
)");
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos) << *result;
    EXPECT_NE(result->find("\"ri_jk_mode\""), std::string::npos) << *result;

    // The physics of the channel pair: one alpha electron and no beta, so the
    // hydrogen carries the whole spin and the beta density is exactly the zero
    // the floor exists for.
    const auto populationsPos = result->find("\"populations\"");
    ASSERT_NE(populationsPos, std::string::npos) << *result;
    const auto mullikenPos = result->find("\"mulliken\"", populationsPos);
    ASSERT_NE(mullikenPos, std::string::npos) << *result;
    const auto mullikenSpin = JsonNumberArray(*result, "\"spin\":", mullikenPos);
    ASSERT_TRUE(mullikenSpin.has_value()) << *result;
    ASSERT_EQ(mullikenSpin->values.size(), 1u);
    EXPECT_NEAR(mullikenSpin->values[0], 1.0, 1e-6);

    // And the energy is the direct family's own H-atom UHF value, because this
    // deck's aux fit has nothing to approximate (one occupied orbital spanned
    // by one basis function, and an aux set far larger than it), so the two
    // families must agree here to the SCF gates - a second, independent
    // reading of the seam's per-spin assembly, on a deck where the beta
    // channel is empty. MEASURED on this fixture at the time of landing:
    // -0.4665818503784859 (ri_jk) against -0.4665818503784860 (direct), 1.1e-16
    // apart - i.e. the fit really is exact here and the 1e-8 band is the loops'
    // own gate, not a fit tolerance.
    auto exact = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 2
atoms = [["H", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_TRUE(exact.has_value()) << exact.error().message;
    const auto exactEnergy = JsonNumber(*exact, "\"total_energy_hartree\"");
    const auto riJkEnergy = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(exactEnergy.has_value());
    ASSERT_TRUE(riJkEnergy.has_value());
    std::printf("UHF H atom/STO-3G (kNormal): ri_jk = %.15e, direct = %.15e, difference = "
                "%.3e\n",
                *riJkEnergy,
                *exactEnergy,
                *riJkEnergy - *exactEnergy);
    EXPECT_NEAR(*riJkEnergy, *exactEnergy, 1e-8)
        << "ri_jk " << *riJkEnergy << ", direct " << *exactEnergy;
}

TEST(DriverErrorTest, RejectsUnwiredCombinations) {
    // The ri_jk cell that stood here is GONE, and it is worth saying why rather
    // than leaving a hole. It asserted kUnimplemented with the message "not
    // wired in v1 (no full-RI exchange path exists)". The composed full-RI builder answered that
    // message's first clause by landing the composed full-RI builder and its run
    // path, so ri_jk is no longer an UNWIRED combination and does not belong in
    // this test at all. Its replacements are all in DriverPinTest: the run
    // (RiJKWiresTheComposedFullRiBuilderEndToEnd), the quality ruling's aux-quality refusal
    // (RiJkRefusesAnAuxThatIsNotAJkFit), and the UHF and Kohn-Sham legs that the
    // old refusal used to SHADOW - it stood first in ValidateCombination, so
    // those whitelists never had to cover this kind before the full-RI builder landed
    // (UhfRiJkWiresThePerSpinAdapterEndToEnd, and for the Kohn-Sham leg
    // RksRiJkRunsTheComposedFullRiCompositionAndDisclosesIt).
    //
    // The sto-3g fixture was changed under the cell by a second, independent
    // commit: The aux-auto rule's universal aux fallback (5b21a295) removed the "no
    // auto-selection rule" refusal, so even that shape now resolves
    // def2-universal-jkfit and runs. Nothing was left for the cell to hold.

    // The UHF + ri_j_link cell that stood here is GONE: it asserted
    // kUnimplemented with the whitelist's "the per-spin adapters of the
    // other builders are untested", and this wiring landed that family's
    // adapter - so the cell's own subject is the run path now, and it lives
    // in DriverPinTest (O2UhfRiJLinkWiresThePerSpinAdapterEndToEnd and
    // UhfRiJLinkMatchesTheRestrictedLegOnAClosedShellDeck), where it can
    // assert what RAN rather than what refused. What replaces it here is the
    // part of that family this wiring did NOT wire: its disk rungs, whose
    // per-spin adapter does not exist. The refusal is up front and by name
    // (never a dropped rung request - the disclosure rule), for the explicit family word;
    // the ladder's own reason keeps a NO-KEY run off the tier when the same
    // keys are named.
    const auto uhfRiJLinkDisk = RunInputText(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["O", 0.0, 0.0, 1.2]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "ri_j_link"
ri_tensor_mode = "disk"
accuracy = "kNormal"
)");
    ASSERT_FALSE(uhfRiJLinkDisk.has_value());
    EXPECT_EQ(uhfRiJLinkDisk.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(uhfRiJLinkDisk.error().message.find("no per-spin adapter"), std::string::npos)
        << uhfRiJLinkDisk.error().message;

    // The forced-disk diagnostic on the same leg: the same missing adapter,
    // refused through the same text - the two keys are two spellings of one
    // request, and a run that dropped either would be the silent
    // substitution the refusal exists to prevent.
    const auto uhfRiJLinkForced = RunInputText(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["O", 0.0, 0.0, 1.2]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
[diagnostics]
force_disk_ri = true
)");
    ASSERT_FALSE(uhfRiJLinkForced.has_value());
    EXPECT_EQ(uhfRiJLinkForced.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(uhfRiJLinkForced.error().message.find("no per-spin adapter"), std::string::npos)
        << uhfRiJLinkForced.error().message;

    // The ri_jk family's cell is NOT this one, and both of its own legs RUN:
    // the pair entry point exposes the halves and each leg has its adapter, so
    // they are pinned next door - the per-spin one by
    // DriverPinTest.UhfRiJkWiresThePerSpinAdapterEndToEnd, and the Kohn-Sham
    // one, which the old Kohn-Sham refusal used to hold closed, by
    // RksRiJkRunsTheComposedFullRiCompositionAndDisclosesIt below.

    // The Kohn-Sham families used to be refused here, and the refusal was
    // load-bearing: the run path selects its method by equality, so an
    // unwired method name did not fail - it ran Hartree-Fock and reported the
    // energy under a DFT label. landed the composition
    // (driver/src/internal/ks_composition.hpp) and the two words now
    // classify, so what this test keeps is the OTHER half of the same
    // posture: the combination refusals that still stand. The end-to-end
    // Kohn-Sham runs live in DriverPinTest (RksSlaterH2.../UksSlaterH2...).
    //
    // The Kohn-Sham pair no longer refuses as a pair, so the naming pin for
    // it moved to DriverErrorTest's
    // UnclassifiedMethodIsRefusedByTheDispatchsSingleCheck; what is left here
    // is the request that reaches a Kohn-Sham word with no functional.

    // The lean spelling is no longer a UHF rejection (the unrestricted seam
    // gave RunDirectUhfScf its lean arm): a UHF + lean request now resolves
    // like the RHF one, and O2UhfExplicitLeanWordWiresTheLeanMember runs
    // it. The rest of this test keeps the combinations that DO refuse.

    // The SAD atomic-multiplicity table stops at Z = 10.
    const auto sadArgon = RunInputText(R"(
[molecule]
atoms = [["Ar", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"
[guess]
type = "sad"
)");
    ASSERT_FALSE(sadArgon.has_value());
    EXPECT_EQ(sadArgon.error().code, qcx::ErrorCode::kUnimplemented);

    // SAD needs the per-element atomic fragment inputs, which only the UHF
    // path builds in v1 (the guess default wired gwh for RHF; sad stays UHF-only); the
    // RHF+sad combination is rejected rather than silently running core.
    const auto rhfSad = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
[guess]
type = "sad"
)");
    ASSERT_FALSE(rhfSad.has_value());
    EXPECT_EQ(rhfSad.error().code, qcx::ErrorCode::kUnimplemented);

    // The hirshfeld promolecular densities need the same SAD
    // atomic-multiplicity table; a heavier element fails the analysis even
    // on an RHF run whose own guess is core.
    const auto hirshfeldArgon = RunInputText(R"(
[molecule]
atoms = [["Ar", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
[properties]
hirshfeld = true
)");
    ASSERT_FALSE(hirshfeldArgon.has_value());
    EXPECT_EQ(hirshfeldArgon.error().code, qcx::ErrorCode::kUnimplemented);
}

TEST(DriverErrorTest, RksRiJkRunsTheComposedFullRiCompositionAndDisclosesIt) {
    // The ri_jk opt-in's last REFUSED cell became a RUN on 2026-09-16, and this
    // row is that cell's new home - the same displacement this test's UHF arm
    // went through one increment earlier, and by the same rule: a refusal cell
    // cannot assert a run, and the run's cell asserts strictly more than the
    // refusal it displaced. The UHF arm asserted kUnimplemented with "UHF is
    // wired with fock_builder" for `type = "uhf"`, which was true while the
    // composed full-RI builder exposed no half; the pair entry point
    // (RiFullFockBuilder::BuildFockHalves) and the per-spin adapter landed, so
    // that leg RUNS - pinned by DriverPinTest.UhfRiJkWiresThePerSpinAdapterEndToEnd.
    //
    // THE DECK BELOW IS THE REFUSAL'S OWN, verbatim (H2/def2-SVP, rks, pbe,
    // kNormal): it asserted kUnimplemented with "the pair is BARE" while the
    // composed full-RI pair had no Kohn-Sham arm - the pair hands out bare
    // J_RI(rho) (no factor of two) and bare K_RI(rho) (no core Hamiltonian), so
    // both of the composition's halves had to be built from it. The arm landed
    // (driver MakeRiFullKsHalf, both Kohn-Sham lanes): H is added to BOTH halves
    // and the Coulomb one is doubled, which is the composition's own contract
    // (H + 2 J(rho) / H - K(rho), internal/ks_composition.hpp's table). So the
    // deck runs, and what is pinned is the RUN.
    //
    // THE ENERGY IS THE ACCOUNTING'S PIN, which is why it is asserted here
    // rather than left to a shape check: the composition contracts J[D] from
    // the SAME halves the Fock was built from, and getting the conversion wrong
    // - H added to one half only (a whole core Hamiltonian), the factor of two
    // left off the Coulomb half (an entire J) - moves this number by O(1) Ha
    // while every Fock that comes out of either mistake still looks plausible.
    // The pinned value also states WHICH method ran: an ri_jk request that fell
    // through to the shared Hartree-Fock tail would report a Hartree-Fock
    // energy under this record's label, and that number is ~0.01 Ha away.
    if (!qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "the JK-fit aux f shells exceed this build's kMaxEngineL (CI lmax=2)";
    }

    const auto rks = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "def2-svp"
[method]
type = "rks"
functional = "pbe"
fock_builder = "ri_jk"
accuracy = "kNormal"
)");
    ASSERT_TRUE(rks.has_value()) << rks.error().message;
    EXPECT_NE(rks->find("\"converged\": true"), std::string::npos);

    // The family and the member word (the ruling: the contraction form, not the
    // family restated), exactly as the Hartree-Fock ri_jk row pins them.
    EXPECT_NE(rks->find("\"builder\": \"ri_jk\""), std::string::npos);
    EXPECT_NE(rks->find("\"builder_member\": \"occ_ri_k\""), std::string::npos);
    EXPECT_NE(rks->find("\"explicit_builder\": \"ri_jk\""), std::string::npos);

    // The disclosure, which is the half of this cell the RULE requires: the
    // exchange half is contracted through the auxiliary fit, so the run states
    // it and states the error it is to be read against,
    // and a consumer must never read this energy as one built from exact
    // quartets. The engine's own rung decision rides beside it .
    EXPECT_NE(rks->find("\"exchange\": \"occ_ri_k\""), std::string::npos);
    EXPECT_NE(rks->find("\"aux_basis\": \"def2-universal-jkfit\""), std::string::npos);
    EXPECT_NE(rks->find("\"exchange_error\""), std::string::npos);
    EXPECT_NE(rks->find("\"ri_jk_mode\""), std::string::npos);

    // The accounting, pinned at the measured value (this build, 2026-09-16).
    const auto total = JsonNumber(*rks, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, -1.115097045817143, 1e-9);
}

TEST(DriverErrorTest, UnclassifiedMethodIsRefusedByTheDispatchsSingleCheck) {
    // The structural half of the unwired-method guard. The dispatch no longer
    // tests `method == kUhf`: RunDriver resolves the word through
    // ResolveScfPath, the ONE exhaustive switch over MethodType, and refuses
    // whatever that switch does not classify. The equality test it replaced
    // answered false for every word nobody had thought about yet and the
    // restricted branch ran on false - so a method this build cannot run
    // produced a Hartree-Fock energy under the input's own label, with nothing
    // downstream able to tell. Refusing at all is the assertion that matters
    // here: no result comes back, so no energy can be reported under a label.
    //
    // The raw value stands for the enumerator a later slice adds. Forcing an
    // unclassified word through the driver is only spellable as an out-of-range
    // value, because the enum deliberately names no unwired member - a value
    // that IS an enumerator without a case arm fails the BUILD instead (the
    // switch's diagnostic guard in run_driver.cpp), which is the compile-time
    // half of the same guarantee.
    auto input = ParseRunInput(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_TRUE(input.has_value()) << input.error().message;
    // The cast is the row's whole point (the rationale above): the value is
    // deliberately not an enumerator, so the range check is suppressed here.
    //
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    input->method.method = static_cast<qcx::io::MethodType>(7);

    const auto result = RunDriver(*input);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("value 7"), std::string::npos);
    EXPECT_NE(result.error().message.find("\"rhf\", \"uhf\", \"rks\" and \"uks\""),
              std::string::npos);

    // A refusal NAMES what it refuses, and the Kohn-Sham pair no longer
    // refuses: it classifies the Kohn-Sham composition and runs. The pin therefore
    // moved to the refusal that still guards the same defect from the other
    // side - a Kohn-Sham word with no `functional` is refused rather than
    // defaulted, because a functional the input did not name is an energy
    // under a label nobody asked for. The message must name the key, the
    // reason and the shipped set, so a refusal that stopped naming any of
    // them is the regression this assertion exists to catch.
    const auto rks = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rks"
accuracy = "kNormal"
)");
    ASSERT_FALSE(rks.has_value());
    EXPECT_EQ(rks.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(rks.error().message.find("needs method.functional"), std::string::npos);
    EXPECT_NE(rks.error().message.find("Shipped: "), std::string::npos);
}

// Kohn-Sham reach test, and the evidence that licenses the removal of
// the dispatch's Kohn-Sham refusal. The removal rests on one claim: an
// `rks`/`uks` request now resolves to a Kohn-Sham path and is run by the
// Kohn-Sham composition. The numerics cannot carry that claim on their own -
// a fall-through to Hartree-Fock produces a plausible energy too, and the
// pre-dispatch produced exactly that, an HF energy under a DFT label.
// The probes below are therefore deterministic and non-numeric: refusals that
// only Kohn-Sham code can produce, each of which an HF fall-through walks
// straight past. Every leg below asserts on the message, never on a code
// alone, because kUnimplemented and kInvalidArgument are shared with the
// refusals the HF path would raise.
//
// Positive control (the red half; recorded with this wiring): with
// ResolveScfPath's two Kohn-Sham cases reverted to kRestricted/kUnrestricted -
// the exact pre-dispatch, which selected its method by equality - all
// four legs RUN instead of refusing, every ASSERT_FALSE below fails, and this
// test is red. Green here is therefore a statement about the path and not
// about the build.
TEST(DriverErrorTest, KohnShamRequestsReachTheKohnShamPath) {
    // Legs 1 and 2 - one per Kohn-Sham case label. `functional` is optional in
    // the schema and refused in the driver, and that refusal lives in the
    // Kohn-Sham context resolver: reaching it means the request classified as
    // Kohn-Sham (not restricted, not unrestricted-plain) and nothing before
    // ResolveKsContext took it away. Two legs rather than one because the two
    // words enter through two different case labels, and a dispatch that wired
    // only one of them would still be the silent-substitution defect on the
    // other.
    const auto rksNaked = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rks"
accuracy = "kNormal"
)");
    ASSERT_FALSE(rksNaked.has_value());
    EXPECT_EQ(rksNaked.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(rksNaked.error().message.find("needs method.functional"), std::string::npos);
    EXPECT_NE(rksNaked.error().message.find("Shipped: "), std::string::npos);

    const auto uksNaked = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "uks"
accuracy = "kNormal"
)");
    ASSERT_FALSE(uksNaked.has_value());
    EXPECT_EQ(uksNaked.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(uksNaked.error().message.find("needs method.functional"), std::string::npos);
    EXPECT_NE(uksNaked.error().message.find("Shipped: "), std::string::npos);

    // Leg 3 - the whitelist, the other refusal that exists only because
    // the request is Kohn-Sham. A family that fuses J and K inside one
    // BuildFock leaves the energy seam no J[D] to contract, so it is refused
    // BY NAME with the wiring rule spelled out. gpu is the family named here
    // because it is the one that STILL fuses: one screening pass, one batch
    // partition, one kernel launch, with J and K atomicAdd-ing into the same
    // buffer, and the CUDA arm is not testable on this machine - so the
    // refusal is the honest answer rather than a composition opened blind.
    // The assertion is the Kohn-Sham sentence and not the error code for the
    // same reason: kUnimplemented is shared with refusals the HF path raises
    // on its own.
    //
    // qfmm was the family named here until 2026-09-16, and ri_jk after it, and
    // the reason each was chosen is the reason each had to be replaced: both
    // run for a Hartree-Fock run on this fixture, and both run for a
    // Kohn-Sham one now too (their compositions landed). The leg moves to the
    // family that still does not.
    const auto wrongFamily = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rks"
functional = "slater"
fock_builder = "gpu"
accuracy = "kNormal"
)");
    ASSERT_FALSE(wrongFamily.has_value());
    EXPECT_EQ(wrongFamily.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(wrongFamily.error().message.find("a Kohn-Sham run (method.type = \"rks\"/\"uks\")"),
              std::string::npos);
    // The admitted set, named - which is the actionable half of the refusal.
    // DATED: this read "fock_builder = \"direct\" only" until the RI-J link's
    // composition landed, then "direct" and "ri_j_link" until the composed
    // QFMM builder's own composition did, and then "ri_j_link" up to the
    // composed full-RI pair's; the sentence below is the same refusal's, with
    // the set it names moved to the families that really do expose the halves
    // the seam contracts.
    EXPECT_NE(wrongFamily.error().message.find("fock_builder = \"direct\", \"qfmm\", "
                                               "\"ri_j_link\" and \"ri_jk\""),
              std::string::npos);

    // Leg 4 - the fukui guard, which is about the analysis and not the Fock:
    // the N +- 1 species run on the direct-UHF runner, so a Kohn-Sham parent
    // would mix two physics under one label. Refused by name, and a
    // Hartree-Fock fall-through runs the analysis happily - the same
    // silent-substitution shape one layer up.
    const auto fukui = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rks"
functional = "slater"
accuracy = "kNormal"
[properties]
fukui = true
)");
    ASSERT_FALSE(fukui.has_value());
    EXPECT_EQ(fukui.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(fukui.error().message.find("properties.fukui is not wired for a Kohn-Sham run"),
              std::string::npos);
}

TEST(DriverErrorTest, GpuSplitRequestIsRefusedByNameNotRunAsDirect) {
    // gpu_split has no [method] word (only ToString knows it, parse_input.cpp
    // :353) and no run path: the io enum's own note says the kind "exists so
    // the driver's selection vocabulary can name the candidate and report it,
    // while execution stays on the wired kinds". A programmatic caller is
    // therefore the only way to request it - the same reachability class as
    // the out-of-range method value the dispatch lane measured completing.
    //
    // Measured before the fix landed (the pre-fix binary is kept beside this
    // this wiring's artifacts): the call returned a COMPLETE document whose
    // resources_resolved.selection.builder, .builder_member and
    // .explicit_builder, and the selection record, all said "gpu_split", while
    // certified_bound.calls = 2 (the collector only the machinery members fold
    // into) and a bit-identical energy against an explicit direct run showed
    // the direct family had done the work - a record contradicting itself,
    // with its primary fields naming a builder that ran nothing. The refusal
    // must name the builder, because the defect is not a missing keyword: the
    // request has no run path at all.
    auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
[resources]
thread_cap = 1
)");
    ASSERT_TRUE(input.has_value()) << input.error().message;
    input->method.builder = qcx::io::BuilderKind::kGpuSplit;

    const auto result = RunDriver(*input);

    ASSERT_FALSE(result.has_value()) << "the request executed instead of being refused";
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(result.error().message.find("gpu_split"), std::string::npos);
    EXPECT_NE(result.error().message.find("no run path"), std::string::npos);
}

TEST(DriverErrorTest, UnnamedBuilderValueIsRefusedRatherThanRunAsDirect) {
    // The runtime half of the machinery tail's admission. The value stands for
    // the enumerator a later slice adds (the method-value technique of the
    // dispatch row above): the resolver honours an explicit builder request
    // verbatim (selection_resolution.cpp), so a kind with no branch would
    // reach WireRhfFockBuilder's shared tail and be built by the direct
    // family while the record named it. The tail's guarded switch stops that
    // for an enumerator at BUILD time; this row pins the value no enumerator
    // names, which no build can catch.
    auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
[resources]
thread_cap = 1
)");
    ASSERT_TRUE(input.has_value()) << input.error().message;
    // The same technique as the row above, for the builder word: the value no
    // enumerator names is the row's subject, so the range check is suppressed.
    //
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    input->method.builder = static_cast<qcx::io::BuilderKind>(99);

    const auto result = RunDriver(*input);

    ASSERT_FALSE(result.has_value()) << "an unnamed builder value executed";
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(result.error().message.find("99"), std::string::npos);
    EXPECT_NE(result.error().message.find("no run path"), std::string::npos);
}

TEST(DriverErrorTest, RefusesALeanRequestPairedWithAFamilyWord) {
    // "lean" is the one spelling that never shares the builder slot with a
    // family word: the parse site writes exactly one of the two
    // (parse_input.cpp - the lean word sets leanDirect and leaves the slot
    // absent), so the pair is reachable only from a PROGRAMMATIC caller.
    // There it would wire the lean builder while the record reported the
    // explicit family word - the exact selector-record disagreement
    // the resolved member name exists to prevent. The driver refuses it up front
    // (the UHF + lean posture), never silently honoring half of it.
    auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_TRUE(input.has_value()) << input.error().message;
    input->method.leanDirect = true;

    const auto result = RunDriver(*input);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("cannot both be requested"), std::string::npos);
    EXPECT_NE(result.error().message.find("fock_builder = \"direct\""), std::string::npos);
    EXPECT_NE(result.error().message.find("fock_builder = \"lean\""), std::string::npos);
}

TEST(DriverErrorTest, LeanWordAdmitsTheLeanMemberAboveTheCeiling) {
    // No fixture in this suite exceeds the lean tier (the C12/C24 QFMM
    // chains are 86 functions), so the above-1000 discrimination is built
    // here from the large-molecule fixture: C50H102 def2-SVP, 1210 basis
    // functions. Above that boundary the ABSENT key resolves to the
    // ri_j_link tier (def2-SVP carries a bundled universal-J aux, so the
    // tier is runnable and no demotion fires) and only the explicit
    // fock_builder = "lean" reaches the direct family's within-family lean
    // member - and each refusal NAMES the family its spelling reached, so
    // the pair IS the above-1000 selector pin. The basis-function count is
    // machine-counted by the product itself (both refusals report
    // n = 1210), never asserted as test arithmetic.
    // What it does NOT cover: a CONVERGED >1000 lean run (its physics and
    // its minutes of SCF cost belong to the ladder harness, not a unit
    // suite). The 1 GiB cap is load-bearing in a second way: a looser one
    // admits the lean builder into a real SCF, so 1 GiB is the measured
    // window where the refusal is clean and fast - below either member's
    // own Create-time requirement (which the refusals themselves report,
    // so no estimate number is pinned here). The window's LOWER edge is no
    // longer a property of this pin: the pre-gate setup admission decides
    // before the ramp, so any cap above the process floor refuses cleanly,
    // and TightCapRefusesBeforeThePreGateSetupRamp holds the crash band to
    // that contract.
    // CAPPED, therefore in the cap child: a memory cap applies to the
    // WHOLE process, so staging it here would leave every test after this
    // one running under the same ceiling when the binary is run directly
    // (cap_child.cpp:1-3; every other cap fixture in this file uses
    // RunCapChild). All three legs run in ONE child under ONE cap - same
    // binary, same baseline, same job object - and the child writes one
    // marker line per leg, so the readings are comparable and each leg's
    // text is checked for the member word its spelling reaches.
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const auto marker = qcx::testing::CapChildMarkerPath(15);
    std::filesystem::remove(marker);

    constexpr double kAboveCeilingCapGib = 1.0;
    const int exitCode =
        qcx::testing::RunCapChild(kAboveCeilingCapGib, marker, "lean_above_ceiling");

    const std::string contents = qcx::testing::ReadCapChildMarker(marker);
    std::filesystem::remove(marker);

    // The marker's line for one leg, its trailing newline stripped.
    const auto legLine = [&contents](const char* legName) {
        const std::string prefix = std::string(legName) + ": ";
        const std::size_t start = contents.find(prefix);

        if (start == std::string::npos)
        {
            return std::string();
        }

        const std::size_t end = contents.find('\n', start);
        return contents.substr(start, end == std::string::npos ? std::string::npos : end - start);
    };

    EXPECT_EQ(exitCode, 1)
        << "all three legs (explicit lean, explicit direct, absent key) must refuse at the "
           "pre-gate setup admission, not complete: "
        << contents;

    // The explicit lean word: the wiring reaches the lean builder at 1210
    // BF, whose own Create-time last-resort check is the admission (Ruling
    // A) and whose refusal names it. The TEXT is the builder's own; the
    // call site is the pre-gate setup admission, which consults the same
    // envelope before the ramp that check used to sit behind (the tight-cap
    // test next door pins the placement, this one the admission band).
    const std::string leanLine = legLine("explicit_lean");
    EXPECT_NE(leanLine.find("refused: "), std::string::npos) << leanLine;
    EXPECT_NE(leanLine.find("lean-direct last-resort memory refusal"), std::string::npos)
        << leanLine;
    EXPECT_NE(leanLine.find("required_peak_bytes"), std::string::npos) << leanLine;

    // The same input with the key ABSENT: under the 2026-09-13 ladder the
    // no-key default at 1210 basis functions is the ri_j_link tier (def2-SVP
    // has a bundled universal-J aux, so the tier is runnable and no demotion
    // fires) - and the lean text above must not appear on this leg. The
    // refusal this leg now carries is the SETUP's, via the pre-gate
    // admission: the family's own budget refusal was the predictive memory
    // model's and went with it (2026-09-17), and the setup - the ramp PLUS
    // the Schwarz screening sweeps the builder runs in front of its own
    // budget decision - is the admission that covers the gap the family's own
    // estimate cannot reach, because that estimate runs behind both.
    const std::string absentLine = legLine("absent_key");
    EXPECT_NE(absentLine.find("refused: "), std::string::npos) << absentLine;
    // This leg's decider, asserted as its own case. The ri_j route answered
    // for itself on 2026-09-18, and it answered in the pre-gate: the
    // ramp-only bound was 483833976 B and this cap is 1 GiB, so the arm
    // correctly ADMITTED this leg and the process then died 0xC0000409 inside
    // the sweeps writing nothing. The bound now charges the sweeps too
    // (internal::SetupPeakBytes), so this leg refuses HERE, before any
    // allocation - the 0.2 s, ~6.5 MiB peak the sweep below measures. The cap
    // stays at 1 GiB: it is the cap that measured the death, and the case it
    // must keep proving is that THIS cap refuses rather than dies.
    EXPECT_NE(absentLine.find("setup ramp needs"), std::string::npos) << absentLine;
    EXPECT_NE(absentLine.find("pre-gate setup ramp"), std::string::npos) << absentLine;
    EXPECT_EQ(absentLine.find("lean-direct"), std::string::npos) << absentLine;
    EXPECT_EQ(absentLine.find("the direct builder models"), std::string::npos) << absentLine;
}

TEST(DriverErrorTest, TightCapRefusesBeforeThePreGateSetupRamp) {
    // The above-ceiling crash class, closed, on the same fixture as the pin
    // next door: at C50H102/def2-SVP the driver's shared setup ramp
    // (BuildCoreHamiltonian and BuildOverlapMatrix) ran BEFORE every
    // admission the run had while each walk materialized the whole
    // contracted pair store (BuildPairMatrix through BuildPairData - ~582
    // MiB of commit on the machinery leg, 942 MiB on the lean leg, the
    // pre-gate measurement at 1210 basis functions), so a cap under
    // that footprint killed the process 0xC0000409 inside the ramp:
    // pre-iteration, empty stdout, empty trace, no mode record, no decision
    // of any kind. The lean, direct and absent-key legs died identically
    // (measured crash band 0.25-0.90 GiB), which is why this is a ramp
    // property and not a selector one.
    //
    // Since 93d5dfc5 each walk builds that pair data CHUNK-WISE (the
    // geometry-only skeleton plus one kPairChunkBytes chunk, released
    // completely per chunk), and this fixture is the case where that
    // changes nothing - which is what keeps the argument below checkable
    // rather than merely old. The store here is 441,172,808 B over 183,921
    // canonical pairs (PairStoreBytes, footprint.hpp; 606 shells, 1210
    // functions), and it is UNDER the 512 MiB cap, so ONE chunk covers the
    // whole store and the chunked walk materializes exactly what the
    // unchunked one did. The cap keeps the same meaning for the same
    // reason; at a size whose store exceeds one chunk the ramp's residency
    // drops instead (C42H86/def2-QZVP, 9.9495 -> 0.7586 GiB per call,
    // MODELLED), and a tighter cap would then be the one to re-derive.
    //
    // 0.5 GiB is INSIDE that band, and that is the test's whole point: the
    // ramp cannot fit 512 MiB at this size at all - its pair data alone is
    // 441,172,808 B and its measured commit is 582.1 MiB - so a run that
    // reaches a refusal here has provably decided before the ramp allocated
    // anything: no timing or allocation-order claim is needed, the
    // arithmetic is the proof. Before the fix this exact invocation exited
    // 0xC0000409 (3221226505) after 1.7-2.8 s with the marker never
    // written; now both legs refuse in ~0.2 s together and the refusal
    // names the placement (the pre-gate setup admission's clause).
    //
    // The child, like every cap fixture in this file: a job-object cap
    // applies to the WHOLE process, so a test that capped its own host
    // binary would kill the suite.
#if defined(QcxHasCuda) || defined(_DEBUG)
    GTEST_SKIP() << "CUDA or Debug build: the load-time footprint of the runtime (~0.25-0.3 "
                    "GiB) or of the Debug exe sits too close to this cap for the child to "
                    "reach the admission; the crash-class contract is verified on the CUDA-less "
                    "Release builds";
#else
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const auto marker = qcx::testing::CapChildMarkerPath(16);
    std::filesystem::remove(marker);

    constexpr double kWithinTheRampCrashBandGiB = 0.5;
    const int exitCode =
        qcx::testing::RunCapChild(kWithinTheRampCrashBandGiB, marker, "lean_above_ceiling");

    const std::string contents = qcx::testing::ReadCapChildMarker(marker);
    std::filesystem::remove(marker);

    // The one assertion the deliverable rests on: a REFUSAL, never the
    // job-object suspension - 3221226505 is 0xC0000409, the fast-fail the
    // ramp used to produce, and a cap-killed child writes no marker at all.
    ASSERT_EQ(exitCode, 1) << "the run must refuse, not die at the cap (0xC0000409 = 3221226505): "
                           << contents;

    const auto legLine = [&contents](const char* legName) {
        const std::string prefix = std::string(legName) + ": ";
        const std::size_t start = contents.find(prefix);

        if (start == std::string::npos)
        {
            return std::string();
        }

        const std::size_t end = contents.find('\n', start);
        return contents.substr(start, end == std::string::npos ? std::string::npos : end - start);
    };

    // Each leg keeps its own admission text (the wiring's, byte for byte)
    // and gains the placement clause: the reader can tell this decision
    // from the same decision taken after the ramp.
    const std::string leanLine = legLine("explicit_lean");
    EXPECT_NE(leanLine.find("refused: "), std::string::npos) << leanLine;
    EXPECT_NE(leanLine.find("lean-direct last-resort memory refusal"), std::string::npos)
        << leanLine;
    EXPECT_NE(leanLine.find("pre-gate setup ramp"), std::string::npos) << leanLine;

    // The explicit "direct" spelling: the third leg of the measured crash
    // (0xC0000409 at the same caps), and the one that makes this a RAMP
    // property rather than a selector one. It now refuses at the ramp, the
    // same admission the absent key meets, carrying the same placement
    // clause - the memory model's own text for this leg went with the model
    // (2026-09-17) and the ramp is what actually stops the run here.
    const std::string directLine = legLine("explicit_direct");
    EXPECT_NE(directLine.find("refused: "), std::string::npos) << directLine;
    EXPECT_NE(directLine.find("setup ramp needs"), std::string::npos) << directLine;
    EXPECT_NE(directLine.find("pre-gate setup ramp"), std::string::npos) << directLine;

    // The absent key at 1210 BF resolves to the ri_j_link tier under the
    // ladder (the pin next door holds the discrimination), and this leg is
    // the one that MEASURED the family's ramp-side gap: without a pre-ramp
    // arm it entered the ramp on the ri_j route and died 0xC0000409 - the
    // same fast-fail the explicit legs used to produce, on the DEFAULT
    // route. The ri_j budget arm that closed it was the memory model's and
    // went with the model; the ramp arm closes it again, one step earlier
    // (the ramp is what every route pays first), so this leg refuses in the
    // ramp's own text.
    const std::string absentLine = legLine("absent_key");
    EXPECT_NE(absentLine.find("refused: "), std::string::npos) << absentLine;
    EXPECT_NE(absentLine.find("setup ramp needs"), std::string::npos) << absentLine;
    EXPECT_NE(absentLine.find("pre-gate setup ramp"), std::string::npos) << absentLine;
    EXPECT_EQ(absentLine.find("lean-direct"), std::string::npos) << absentLine;
#endif
}

TEST(DriverErrorTest, RejectsBadQfmmSchemaKnobsByName) {
    // The schema gates: a
    // bad theta/l_mult/max_leaf_size/crossover on the qfmm path is
    // rejected through the full input pipeline with the error naming the
    // schema key - the parser applies the engine's Create() contract up
    // front, so no run starts with a knob the builder would refuse or
    // silently bend.
    struct Case {
        const char* text;
        const char* messageFragment;
    };

    const Case cases[] = {
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.74]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kNormal\"\n"
         "theta = nan\n",
         "method.theta"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.74]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kNormal\"\n"
         "l_mult = 9\n",
         "method.l_mult"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.74]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kNormal\"\n"
         "max_leaf_size = 0\n",
         "method.max_leaf_size"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.74]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kNormal\"\n"
         "crossover_basis_function_count = -1\n",
         "method.crossover_basis_function_count"},
    };

    for (const auto& test : cases)
    {
        const auto result = RunInputText(test.text);
        ASSERT_FALSE(result.has_value()) << "case: " << test.text;
        EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(result.error().message.find(test.messageFragment), std::string::npos)
            << "message: " << result.error().message;
    }
}

// The key split (owner-ruled 2026-09-13) on the four QFMM tuning knobs: a
// key that names a MECHANISM the author is relying on is refused by name on a
// family that has no route for it, never accepted and dropped. Before this
// rule the knob was parsed, validated and stored, and then read by nothing
// outside the two qfmm arms (RunQfmmUhfScf and WireRhfFockBuilder's kQfmm
// branch), so a run of any other family went ahead at the engine's preset
// while the document asked for a screening it never got - the #24 defect
// class: an unwired request running something else under the label it carried.
// theta is the one to keep in view, because it is an accuracy control and not
// a performance knob (integrals/src/qfmm_fock_build.cpp decides which pairs
// the near/far split computes exactly from it).
//
// Two boundaries are this row's own, because the ruling turns on both:
//   - the key PRESENT refuses and the key ABSENT runs (the default arm: an absent
//     key is the system's judgement, and an absent QFMM knob keeps the
//     engine's preset default) - legs 1 and 4;
//   - the refusal names the family as the FILE stated it: an explicit word is
//     quoted back, and a run that wrote no family word is not described as
//     though it had - legs 1, 2 and 3. The absent-key spelling is the size
//     ladder's, never a family name: since the 2026-09-13 ruling an absent
//     fock_builder resolves THROUGH the ladder, so a refusal saying "the
//     direct family" would name a resolution this run never made (it is the
//     ladder's bottom tier at this fixture size, and saying otherwise is the
//     silent substitution the disclosure rule forbids).
TEST(DriverErrorTest, QfmmKnobsOnAnotherFamilyAreRefusedByName) {
    struct Case {
        const char* text;
        const char* messageFragment;
    };

    // Leg 1 - one refusal per knob, each on the family an ABSENT
    // fock_builder resolves to at this fixture size (the 2026-09-13
    // ladder's bottom tier: the direct family lean member at these H2-class
    // sizes), with the absent-key spelling of the ladder text asserted on
    // every one of them.
    const Case cases[] = {
        {R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
theta = 0.5
)",
         "method.theta is set on a run whose fock_builder is not \"qfmm\""},
        {R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
l_mult = 8
)",
         "method.l_mult is set on a run whose fock_builder is not \"qfmm\""},
        {R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
max_leaf_size = 4
)",
         "method.max_leaf_size is set on a run whose fock_builder is not \"qfmm\""},
        {R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
crossover_basis_function_count = 1000
)",
         "method.crossover_basis_function_count is set on a run whose fock_builder is not "
         "\"qfmm\""},
    };

    for (const auto& test : cases)
    {
        const auto result = RunInputText(test.text);
        ASSERT_FALSE(result.has_value()) << "the knob was accepted and dropped: " << test.text;
        EXPECT_EQ(result.error().code, qcx::ErrorCode::kUnimplemented);
        EXPECT_NE(result.error().message.find(test.messageFragment), std::string::npos)
            << "message: " << result.error().message;
        // The ladder's spelling of the absent-key case, re-pinned 2026-09-14:
        // this assertion pinned "the direct family (no fock_builder was
        // given" until the ladder landed (afca4a2e, 2026-09-13), and that
        // text is GONE from FamilyNameForRefusal by design - the ladder
        // governs the absent case, so the refusal names the ladder tier the
        // run is on and leaves the family word to the legs that wrote one.
        // The bound above is unchanged: an absent key still refuses the knob
        // and still names the key the file wrote.
        EXPECT_NE(result.error().message.find("this run is on the no-fock_builder size ladder's "
                                              "own tier"),
                  std::string::npos)
            << "message: " << result.error().message;
        EXPECT_NE(result.error().message.find("Use fock_builder = \"qfmm\", or remove the key"),
                  std::string::npos)
            << "message: " << result.error().message;
    }

    // Leg 2 - the explicit family word, quoted back, with TWO knobs present:
    // the refusal names the keys the file wrote (both of them, in schema
    // order) and never reads as though no family word had been given.
    const auto explicitWord = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
theta = 0.5
max_leaf_size = 4
)");
    ASSERT_FALSE(explicitWord.has_value()) << "the knobs were accepted and dropped";
    EXPECT_EQ(explicitWord.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(explicitWord.error().message.find("method.theta and method.max_leaf_size are set"),
              std::string::npos)
        << "message: " << explicitWord.error().message;
    EXPECT_NE(explicitWord.error().message.find("this run is on fock_builder = \"direct\""),
              std::string::npos)
        << "message: " << explicitWord.error().message;
    EXPECT_NE(explicitWord.error().message.find("or remove the keys"), std::string::npos)
        << "message: " << explicitWord.error().message;

    // Leg 3 - the lean word, which is a within-family request and never
    // occupies the builder slot (the parse site sets leanDirect). The family
    // text therefore has to name the family AND the member: "no fock_builder
    // was given" would be false here, and a refusal that misstates the
    // document cannot be acted on.
    const auto leanWord = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "lean"
accuracy = "kNormal"
theta = 0.5
)");
    ASSERT_FALSE(leanWord.has_value()) << "the knob was accepted and dropped on the lean member";
    EXPECT_EQ(leanWord.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(leanWord.error().message.find("this run is on the direct family "
                                            "(fock_builder = \"lean\""),
              std::string::npos)
        << "message: " << leanWord.error().message;

    // Leg 4 - the ABSENT key, the other half of the ruling: the same fixture
    // with no knob runs. This is the leg that fails if the rule is written as
    // "the family is not qfmm, refuse" without asking whether the key is
    // there at all.
    const auto absentKey = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
)");
    ASSERT_TRUE(absentKey.has_value()) << absentKey.error().message;
    EXPECT_NE(absentKey->find("\"converged\": true"), std::string::npos);

    // Leg 5 - the family the knob belongs to still takes it: the same key on
    // fock_builder = "qfmm" is NOT refused. The numbers are the
    // passthrough pin's (H2QfmmRhfWithSchemaKnobsRunsToConvergence above);
    // this leg owns the boundary only, so that a rule which refused the knob
    // on every family could not pass by refusing the right one too.
    const auto qfmmFamily = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "qfmm"
accuracy = "kNormal"
theta = -1.0
)");
    ASSERT_TRUE(qfmmFamily.has_value()) << qfmmFamily.error().message;
    EXPECT_NE(qfmmFamily->find("\"converged\": true"), std::string::npos);
}

TEST(DriverRunTest, TheQfmmRunRecordDisclosesTheModelThatRan) {
    // The disclosure rule disclosure the 2026-09-15 default flip created .
    // The engine could state which model a build ran (QfmmJBuilder::
    // ModelRecord) and the run record carried none of it, so the default
    // behaviour of every QFMM run changed while the document a consumer reads
    // said nothing about it - the class the driver's own knob rule calls "an
    // unwired request running something else under the label it carried".
    // These legs are the states a reader must be able to tell apart: the
    // default aim, the written angle that overrides it, and the gate that
    // empties the far field. This is the one place the mode word is itself the
    // fact under test rather than a proxy for one - the subject is which model
    // the record NAMES, which no assertion about a count can establish - so
    // the word stays, with the run's own far-pair count beside it as the
    // quantity that named model admitted.
    //
    // Leg 1 - the default: no key writes a model, and the record says which
    // one ran. The extent is the product-distribution ball and the separation
    // test the centre-to-width form, aimed at the preset's OWN angle - 0.3 at
    // kNormal - because an absent theta means "not given" and resolves to the
    // rung's number rather than to one global one. separation_k reads 0.0
    // because the surface test is not what separated the tree: the engine
    // records the parameter of the test that RAN and zeroes the other one.
    const auto defaultRun = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "qfmm"
accuracy = "kNormal"
)");
    ASSERT_TRUE(defaultRun.has_value()) << defaultRun.error().message;
    EXPECT_NE(defaultRun->find("\"qfmm_model\""), std::string::npos);
    EXPECT_NE(defaultRun->find("\"extent_model\": \"kProductBall\""), std::string::npos);
    EXPECT_NE(defaultRun->find("\"separation_mode\": \"kWidthTheta\""), std::string::npos);
    EXPECT_NE(defaultRun->find("\"theta\": 0.3"), std::string::npos);
    EXPECT_NE(defaultRun->find("\"separation_k\": 0.0"), std::string::npos);
    // The run's own far field: two atoms admit no far pair at any aim, so 0 is
    // the honest count here - the claim that the far field CAN be nonzero
    // rides the chain pins, where the count moves with the preset.
    EXPECT_NE(defaultRun->find("\"geometric_far_pair_count\": 0,"), std::string::npos);

    // Leg 2 - the record follows the BUILD, not the default. An explicit
    // theta > 0 overrides the ANGLE the default test classifies with (the
    // historic contract: no written accuracy control is silently ignored), so
    // the same fixture with theta = 0.5 must come back at that number rather
    // than at the preset's 0.3, and separation_k must still read the 0.0 of
    // the test that did not run. A record that echoed the default passes leg 1
    // and fails here.
    const auto widthTest = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "qfmm"
accuracy = "kNormal"
theta = 0.5
)");
    ASSERT_TRUE(widthTest.has_value()) << widthTest.error().message;
    EXPECT_NE(widthTest->find("\"separation_mode\": \"kWidthTheta\""), std::string::npos);
    EXPECT_NE(widthTest->find("\"theta\": 0.5"), std::string::npos);
    EXPECT_NE(widthTest->find("\"separation_k\": 0.0"), std::string::npos);

    // Leg 3 - the absence rule: a family that builds no QFMM Coulomb half has
    // no model to report, and the block is ABSENT rather than defaulted. A
    // defaulted block would attribute the flipped model to a run that never
    // built an octree, which is the fabrication the block exists to prevent.
    const auto directRun = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_TRUE(directRun.has_value()) << directRun.error().message;
    EXPECT_EQ(directRun->find("\"qfmm_model\""), std::string::npos);

    // Leg 4 - the two words are INDEPENDENT facts, which is the whole reason
    // the block carries two members instead of one "model" word. Leg 2's
    // explicit theta moved the separation ANGLE and must NOT have moved the
    // extent model: the same product-ball geometry ran under both aims, and
    // a reader comparing the two runs must see exactly one member change. The
    // retired extent word appears in neither document - it is not selectable
    // from any input key, so a run that reported it would be reporting a
    // model nothing could have asked for.
    EXPECT_NE(widthTest->find("\"extent_model\": \"kProductBall\""), std::string::npos);
    EXPECT_EQ(defaultRun->find("kMidpointBound"), std::string::npos);
    EXPECT_EQ(widthTest->find("kMidpointBound"), std::string::npos);

    // Leg 5 - the degenerate gate, whose DISCLOSURE shape is worth pinning
    // because it is not obvious: theta < 0 does not arrive as a negative
    // number in the record. It resolves the empty far field by driving the
    // classified angle to 0.0 - under the centre-to-width test theta <= 0 is
    // "never well separated", the same gate every absent-theta run resolves to
    // at a rung whose own angle is 0.0 - so the record reads the default mode
    // word with theta 0.0, and the far-pair count states the consequence: the
    // build admitted none. A reader who took a negative theta to mean "the
    // width test ran with a negative angle" would misread this build.
    const auto gate = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "qfmm"
accuracy = "kNormal"
theta = -1.0
)");
    ASSERT_TRUE(gate.has_value()) << gate.error().message;
    EXPECT_NE(gate->find("\"separation_mode\": \"kWidthTheta\""), std::string::npos);
    EXPECT_NE(gate->find("\"theta\": 0.0"), std::string::npos);
    EXPECT_NE(gate->find("\"separation_k\": 0.0"), std::string::npos);
    EXPECT_NE(gate->find("\"geometric_far_pair_count\": 0,"), std::string::npos);
}

TEST(DriverPinTest, H2QfmmRhfWithSchemaKnobsRunsToConvergence) {
    // The positive side of the passthrough: parse-valid knobs reach
    // QfmmOptions through the driver and the QFMM run completes. The
    // kQfmm branch wires the composed builder composed builder (F = H + 2J_QFMM - K),
    // so the run is physical RHF: at the theta < 0 degenerate gate the
    // composed Fock equals the fused direct one (up to the split-pass fp
    // pairing), and the run must reproduce the H2 direct pin. The theta
    // < 0 spelling exercises the degenerate gate explicitly, l_mult 8 the
    // cap, leaf size 4 and the crossover override the remaining
    // passthroughs.
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const auto result = RunInputText(R"(
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
fock_builder = "qfmm"
accuracy = "kTight"
theta = -1.0
l_mult = 8
max_leaf_size = 4
crossover_basis_function_count = 1000

[scf]
max_iterations = 100
energy_tolerance = 1e-8
density_tolerance = 1e-6
use_diis = true
)");
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

    // Measured 2026-09-16 and VESTIGIAL: the cell's assertions are the schema-knob
    // record facts (theta < 0, l_mult 8, leaf 4 - all gate-independent) plus the
    // energy. At the operating default the record is byte-identical on every one of
    // them: -1.1167143252 within 1e-8, workspace_budget with capacity_bytes
    // 17179845832, mode_record kFastPath, converged true.

    // The composed-QFMM RHF run joined the budget path (the composed builder): the
    // default 16.0 GiB cap becomes a workspace budget of the cap MINUS the
    // run's reserve, shared with
    // both nested halves - the reserve being
    // integrals::SetupAdmissionReserveBytes, the engine's own
    // ramp-and-sweeps bound (SetupRampPeakBytes) times its one slack
    // constant (kSetupAdmissionSlack, 1.5). At this fixture - H2/sto-3g, no
    // aux in effect - the bound is 15,568 B, so the reserve is the 23,352 B
    // above it and the capacity is 16.0 * 2^30 - 23,352 = 17179845832 bytes.
    // (The pre-deletion pin recorded the model's fitted cap-minus-base
    // 17119739641; nothing of that model is restored.) The record also
    // carries a mode_record with the Coulomb
    // (QFMM) half's Create-time decision (the RHF surface's single-
    // record shape; the exchange half's commit is inside the budget's
    // committed_bytes).
    EXPECT_NE(result->find("\"workspace_budget\""), std::string::npos);
    EXPECT_NE(result->find("\"capacity_bytes\": 17179845832"), std::string::npos);
    EXPECT_NE(result->find("\"mode_record\""), std::string::npos);
    EXPECT_NE(result->find("\"mode\": \"kFastPath\""), std::string::npos);

    const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, -1.1167143252, 1e-8);
}

TEST(DriverPinTest, H2oQfmmComposedRhfReproducesThePinAtAllPresets) {
    // The composed-RHF H2O gate (the pin table): the qfmm
    // builder must reproduce the direct kTight pin -74.96292827 within
    // the 1e-5 band at EVERY preset - the kTight rung through the
    // degenerate gate (the composed Fock equals the fused direct one up
    // to the split-pass fp pairing), kLoose/kNormal through their real
    // far-field/lane approximations (each certified far inside the band).
    // The text is kH2oToml's - the direct pin run's own geometry - with
    // only the [method] builder/accuracy tokens retuned.
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const char* presets[] = {"kLoose", "kNormal", "kTight"};

    for (const char* preset : presets)
    {
        const auto toml = WithBuilderAndAccuracy(kH2oToml, "qfmm", preset);
        auto result = RunInputText(toml.c_str());
        ASSERT_TRUE(result.has_value()) << result.error().message;
        EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

        const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
        ASSERT_TRUE(total.has_value());
        std::cout << "H2O qfmm preset " << preset << ": " << *total << "\n";
        EXPECT_NEAR(*total, -74.96292827, 1e-5) << "preset " << preset;
    }
}

TEST(DriverPinTest, C6ChainQfmmRhfMatchesDirectInsideThePresetBudgets) {
    // The composed builder chain pins' kTight rung, on the smallest fixture that carries
    // it (owner fixture-size rule, 2026-09-18: a correctness claim is
    // asserted on the smallest fixture that exercises it, never on a large
    // system; large runs are for performance and memory). The claim is the
    // row the C12 and C24 pins used to carry, unchanged: at kTight the
    // composed builder resolves the degenerate gate (separationK = -1,
    // nothing well separated, FarFieldPairCount() = 0 - the composed Fock
    // IS the restricted near-field direct build with the split pass' H/K
    // bookkeeping), and the composed total must reproduce the same-preset
    // direct to 1e-9, a bound deliberately STRICTER than the re-derived
    // 1e-8 QFMM J-half ladder (QfmmBudgetForPreset): the row compares one
    // Fock through two code paths at an equal preset, so the far-field
    // budget does not bound it and the bound is the row's own acceptance
    // bound.
    //
    // WHY THE SMALLER FIXTURE DOES NOT WEAKEN IT. What the row measures is
    // the split pass' fp agreement, and that mechanism is size-independent
    // by construction - there is no size threshold under which the composed
    // and fused paths stop being the same arithmetic. For scale, the same
    // row measured 1.36424e-12 at C12's 86 functions (2026-09-13). Measured
    // at this fixture (44 functions) on this tree 2026-09-18, at the defaults rule
    // default SCF gate 1e-8/1e-6 that AlkaneChainToml writes: 5.1e-13,
    // ~2000x inside the band. Cost at the same gates: the direct run
    // converges in 12 iterations / ~9 s of SCF wall, the composed run in 12
    // / ~55 s - against the ~5-8 minutes the composed half alone cost at
    // C12 and C24.
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto molecule = qcx::testing::MakeAlkaneSto3g(6);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // The fixture's basis-function total, counted through the framework's
    // own parser - the same basis text the runs below parse, so the count
    // is the runs' n_basis. The size-class band row prints it (the
    // report's bin assignment is machine-counted, never arithmetic).
    const auto chainBasis = qcx::testing::MakeAlkaneSto3gBasis();
    ASSERT_TRUE(chainBasis.has_value()) << chainBasis.error().message;
    const int basisCount =
        static_cast<int>(qcx::driver::CountBasisFunctions(*molecule, *chainBasis));

    const auto directToml = AlkaneChainToml(*molecule, "direct", "kTight");
    auto directResult = RunInputText(directToml.c_str());
    ASSERT_TRUE(directResult.has_value()) << directResult.error().message;
    EXPECT_NE(directResult->find("\"converged\": true"), std::string::npos);
    const auto directTotal = JsonNumber(*directResult, "\"total_energy_hartree\"");
    ASSERT_TRUE(directTotal.has_value());

    const auto qfmmToml = AlkaneChainToml(*molecule, "qfmm", "kTight");
    auto qfmmResult = RunInputText(qfmmToml.c_str());
    ASSERT_TRUE(qfmmResult.has_value()) << qfmmResult.error().message;
    EXPECT_NE(qfmmResult->find("\"converged\": true"), std::string::npos);
    const auto qfmmTotal = JsonNumber(*qfmmResult, "\"total_energy_hartree\"");
    ASSERT_TRUE(qfmmTotal.has_value());

    // The rung is asserted from the run's OWN record, not inferred from the
    // fixture's size: at kTight the composed builder resolves the degenerate
    // gate (its own angle, 0.0, which under the centre-to-width test is
    // "never well separated"), and the record states the consequence as the
    // run's own far-pair count - the build admitted none. A run whose far
    // field was live would carry a nonzero count here, and this row's
    // comparison would no longer be the split pass' alone - which is the
    // reading the row depends on.
    EXPECT_NE(qfmmResult->find("\"theta\": 0.0"), std::string::npos)
        << "the kTight rung's own angle";
    EXPECT_NE(qfmmResult->find("\"geometric_far_pair_count\": 0,"), std::string::npos)
        << "the composed run must record an empty far field";

    const double ktightDifference = std::abs(*qfmmTotal - *directTotal);
    std::cout << "C6 preset kTight: E_qfmm " << *qfmmTotal << ", E_direct-kTight " << *directTotal
              << ", difference " << ktightDifference << "\n";
    EXPECT_LT(ktightDifference, 1e-9) << "preset kTight";

    // The kTight composed row's band line: the degenerate gate's context
    // measures no halves separately, so the line prints the bin and the
    // composed deviation only (the report never invents half claims). The
    // composed run's own record carries the degenerate gate - the far field
    // empty - which is what makes this row's comparison the split pass'
    // alone.
    {
        qcx::driver::QfmmAccuracyRunRow bandRow;
        bandRow.fixture = "C6H14/STO-3G";
        bandRow.basisCount = basisCount;
        bandRow.preset = qcx::integrals::AccuracyPreset::kTight;
        bandRow.reference = "kTight direct";
        bandRow.composedDeviation = ktightDifference;
        std::cout << qcx::driver::FormatSizeClassBandLine(bandRow) << "\n";
        std::cout << qcx::driver::FormatSizeClassBandTable({bandRow}) << "\n";
    }
}

TEST(DriverPinTest, C12ChainQfmmRhfMatchesDirectInsideThePresetBudgets) {
    // The composed builder chain pins on C12H26/STO-3G. The QFMM J-half budgets
    // {1e-5, 1e-8} Eh for {kLoose, kTight} (the accuracy record; the kTight rung re-derived
    // from 1e-9 by the 2026-09-13 ruling - the ladder is
    // QfmmBudgetForPreset, accuracy.hpp)
    // (benchmarks/data/qfmm_ladder_sweep_full.csv) bound the QFMM J half
    // alone: the record's energy errors were measured against the near-exact
    // kNormal ground truth with the direct K half on both sides. The
    // composed Fock's exchange half is the direct builder at the run's own
    // preset screening, so the composed-vs-near-exact deviation at kLoose is
    // dominated by the exchange half's own kLoose class: measured
    // 2.2894033804732317e-05 at C12 on 2026-09-03 - the kLoose DIRECT method
    // itself sits that far from near-exact on this fixture (the
    // discriminator rows below measure and band the class on every gate
    // run). The class is a property of the direct exchange at a 1e-8 density
    // threshold at the 86-function scale, NOT a QFMM defect: the composed
    // path reproduces the kLoose direct to the QFMM scale (~3e-8 at C12).
    // Per the 2026-09-03 ruling, the kLoose row therefore compares the
    // composed run against the near-exact kTight direct (a strict superset
    // of the record's reference; its own deviation is the ~1e-12 threshold
    // scale) at the measured exchange-class budget 1e-4 (~4x the measured
    // class). The kLoose rung's far field is live on this fixture, and the
    // run's own record states it: 2080 far pairs at the rung's own angle 0.45
    // (measured 2026-09-20; the CSV record's 14 pairs at the (0.45, 1e-6)
    // midpoint-extent winner is the same claim under the retired extent
    // model). The kNormal rung's pin lives on C24
    // (C24ChainQfmmRhfMatchesDirectInsideThePresetBudgets) under the
    // fixture-size rule below rather than for vacuity: the accuracy record's C12
    // kNormal cell was all-near (bit-exact by construction - the accuracy record, point 4),
    // but the current default records 517 far pairs there.
    //
    // WHAT THIS FIXTURE IS FOR, and why it is not smaller (owner
    // fixture-size rule, 2026-09-18: a correctness claim is asserted on the
    // smallest fixture that exercises it; large runs are for performance and
    // memory). The kLoose rows below are the only rows whose size is set by
    // an assertion rather than by taste: their discriminator band's 1e-5
    // FLOOR is the exchange class the budget 1e-4 is read against, and the
    // class grows with the basis. Measured on this tree, each at the defaults rule
    // default SCF gate: 2.19498e-6 at C6 (44 functions), 6.9938e-6 at C8
    // (58), 2.29e-5 here (86) - the smallest fixture in the family whose
    // class clears 1e-5. THE kTight ROW IS NOT HERE: its claim - the
    // composed run reproduces the same-preset direct to 1e-9 through the
    // degenerate gate - is size-independent by mechanism (the far field is
    // empty at that preset and the row measures the split pass' fp
    // agreement), so it is asserted at C6, in
    // C6ChainQfmmRhfMatchesDirectInsideThePresetBudgets. The direct kTight
    // run stays here: the kLoose rows below use it as their near-exact
    // reference (and it is the more accurate reference, its own deviation
    // being at the threshold scale rather than the class scale).
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto molecule = qcx::testing::MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // The fixture's basis-function total, counted through the framework's
    // own parser - the same basis text the runs below parse, so the count
    // is the runs' n_basis. The size-class band rows print it (the
    // report's bin assignment is machine-counted, never arithmetic).
    const auto chainBasis = qcx::testing::MakeAlkaneSto3gBasis();
    ASSERT_TRUE(chainBasis.has_value()) << chainBasis.error().message;
    const int basisCount =
        static_cast<int>(qcx::driver::CountBasisFunctions(*molecule, *chainBasis));

    // The measured composed rows of this test, accumulated for the
    // aggregate size-class band table printed at the end .
    std::vector<qcx::driver::QfmmAccuracyRunRow> bandRows;

    // The shared near-exact reference and the discriminator's class run:
    // both direct presets run once and feed every row below.
    const auto ktightDirectToml = AlkaneChainToml(*molecule, "direct", "kTight");
    auto ktightResult = RunInputText(ktightDirectToml.c_str());
    ASSERT_TRUE(ktightResult.has_value()) << ktightResult.error().message;
    EXPECT_NE(ktightResult->find("\"converged\": true"), std::string::npos);
    const auto ktightDirectTotal = JsonNumber(*ktightResult, "\"total_energy_hartree\"");
    ASSERT_TRUE(ktightDirectTotal.has_value());

    const auto klooseDirectToml = AlkaneChainToml(*molecule, "direct", "kLoose");
    auto klooseResult = RunInputText(klooseDirectToml.c_str());
    ASSERT_TRUE(klooseResult.has_value()) << klooseResult.error().message;
    EXPECT_NE(klooseResult->find("\"converged\": true"), std::string::npos);
    const auto klooseDirectTotal = JsonNumber(*klooseResult, "\"total_energy_hartree\"");
    ASSERT_TRUE(klooseDirectTotal.has_value());

    // The discriminator band: the kLoose direct's own deviation from the
    // near-exact kTight direct must sit in (1e-5, 1e-3) on every gate run -
    // that measurement IS the exchange-screening class the 1e-4 kLoose
    // budget below is set from, and the band keeps the attribution live.
    const double looseClassDeviation = std::abs(*klooseDirectTotal - *ktightDirectTotal);
    std::cout << "C12 kLoose direct vs kTight direct: " << looseClassDeviation << "\n";
    EXPECT_GT(looseClassDeviation, 1e-5) << "the kLoose direct's own screening class";
    EXPECT_LT(looseClassDeviation, 1e-3) << "the kLoose direct's own screening class";

    // kLoose row: the composed run vs the near-exact kTight direct at the
    // measured exchange-class budget 1e-4 (the accuracy 1e-5 record bounds the
    // QFMM J half - its C12-kLoose row is the CSV's 1.605e-6 - while the
    // exchange half shares the direct's kLoose class banded above). The
    // same-preset fidelity companion pins the composed to the kLoose direct
    // at the QFMM scale (~3e-8 measured), which is the attribution letting
    // the 1e-4 row be read as the direct's own class, not a QFMM overrun.
    const auto qfmmKlooseToml = AlkaneChainToml(*molecule, "qfmm", "kLoose");
    auto qfmmKlooseResult = RunInputText(qfmmKlooseToml.c_str());
    ASSERT_TRUE(qfmmKlooseResult.has_value()) << qfmmKlooseResult.error().message;
    EXPECT_NE(qfmmKlooseResult->find("\"converged\": true"), std::string::npos);
    const auto qfmmKlooseTotal = JsonNumber(*qfmmKlooseResult, "\"total_energy_hartree\"");
    ASSERT_TRUE(qfmmKlooseTotal.has_value());

    // The kLoose row's rung, from the run's own record: the far field must be
    // LIVE in the run whose number the rows below read. The run states that
    // as its own geometry - the classification angle the kLoose rung aims
    // (0.45) and the far-pair count it admitted (2080 on this fixture,
    // measured 2026-09-20) - rather than as the name of the test that decided
    // it: a count of 0 is the degenerate gate, whatever word the record
    // carries. Without a live far field the composed run would be a
    // near-field build and the kLoose row would assert nothing about the
    // multipoles.
    EXPECT_NE(qfmmKlooseResult->find("\"theta\": 0.45"), std::string::npos)
        << "the kLoose rung's own angle";
    EXPECT_NE(qfmmKlooseResult->find("\"geometric_far_pair_count\": 2080,"), std::string::npos)
        << "the kLoose composed run must record a live far field";

    const double klooseCompositionFidelity = std::abs(*qfmmKlooseTotal - *klooseDirectTotal);
    EXPECT_LT(klooseCompositionFidelity, 1e-5) << "preset kLoose (composition fidelity)";
    const double klooseNearExactDeviation = std::abs(*qfmmKlooseTotal - *ktightDirectTotal);
    std::cout << "C12 preset kLoose: E_qfmm " << *qfmmKlooseTotal << ", E_direct-kTight "
              << *ktightDirectTotal << ", difference " << klooseNearExactDeviation << "\n";
    EXPECT_LT(klooseNearExactDeviation, 1e-4) << "preset kLoose";

    // The size-class band line of this composed row (additive
    // REPORTING - the per-run band is text-only, never a budget, and
    // never enters the result JSON; the constant budgets {1e-5, 1e-7,
    // 1e-9} bound the QFMM Coulomb half alone). The row reports its
    // pre-registered bin, the composed deviation vs the near-exact
    // reference, and - where this run's context measured it - the
    // exchange half's class: at C12 the kLoose direct's discriminator
    // class is exchange-dominated (the attribution above), so the
    // discriminator measurement rides as the exchange half's |dE|.
    {
        qcx::driver::QfmmAccuracyRunRow bandRow;
        bandRow.fixture = "C12H26/STO-3G";
        bandRow.basisCount = basisCount;
        bandRow.preset = qcx::integrals::AccuracyPreset::kLoose;
        bandRow.reference = "kTight direct";
        bandRow.composedDeviation = klooseNearExactDeviation;
        bandRow.exchangeHalfDeviation = looseClassDeviation;
        std::cout << qcx::driver::FormatSizeClassBandLine(bandRow) << "\n";
        bandRows.push_back(bandRow);
    }

    // kTight row: MOVED to C6ChainQfmmRhfMatchesDirectInsideThePresetBudgets
    // (owner fixture-size rule, 2026-09-18 - see this test's header note).
    // The claim and its 1e-9 band are unchanged there, on a fixture whose
    // two runs cost ~9 s and ~55 s against this fixture's ~5-8 minutes for
    // the composed run alone. The near-exact kTight DIRECT run stays above:
    // it is this test's kLoose reference.

    // The aggregate size-class band table over this test's measured rows
    // (the pre-registered-bin schema: n_fixtures, N_basis range,
    // max/median/p95 |dE|, worst fixture; the n < 3 underpopulation
    // marker fires honestly - each (preset, reference) group here holds
    // one fixture, so no fit is claimed).
    std::cout << qcx::driver::FormatSizeClassBandTable(bandRows) << "\n";
}

TEST(DriverPinTest, C24ChainQfmmRhfMatchesDirectInsideThePresetBudgets) {
    // The C24H50/STO-3G half of the composed builder chain pins. C24 carries the
    // NON-vacuous kNormal rung (499 far pairs at the (0.3, 1e-8) winner -
    // the C12 tooth is all-near there; the accuracy record, point 5), the strongest kLoose
    // exercise (1997 far pairs), and the kTight degenerate rung. Budgets
    // and references follow the C12 test's 2026-09-03 ruling, with two
    // C24-specific notes. First, C24 kLoose is where the exchange-class
    // bites hardest: measured 2026-09-03, the kLoose direct sits
    // 1.49943e-4 from the near-exact kTight direct and the composed run
    // 1.29971e-4 - the class (the exchange half's dropped quartet mass at
    // its 1e-8 density screening, shared by every composed run whose
    // exchange half is the direct builder) GROWS with the basis, 2.29e-5
    // at C12's 86 functions to ~1.3-1.5e-4 at C24's 170 (the framework's
    // own parse count, 5 functions per carbon - see the band rows below
    // and the AlkaneChainFixtureCountsAreMachineCounted pin), so no
    // single tight constant serves both fixtures. At C12 the composed run
    // reproduces the kLoose direct to the QFMM scale (3.2e-8), and that is
    // asserted there; at C24 the two deviations were measured to invert, so the
    // relationship between them is REPORTED here and not asserted - the note at
    // the band below carries the reading and why the two are different
    // quantities (the direct's own screened-J far-region drop against the
    // composed path's distance from the kTight direct, which the composed route
    // never promised to shrink; the accuracy record CSV's C24-kLoose J-half record, 3.84e-7
    // at a fixed reference density, is NOT an SCF-level bound: the converged
    // density shifts under the exchange half's screening and carries the SCF
    // deviation up to the class). The kLoose rows therefore assert the C24
    // exchange-class pin 9.1e-4 (1.50x the composed deviation measured
    // 2026-09-19 under the far-field default: C24H50/STO-3G, composed kLoose vs
    // direct kTight, product-ball extent, surface-ball separation k = 1.0, no
    // explicit theta; 6.074926e-4 on three builds within 7.5e-9 relative, and
    // the 2026-09-03 composed reading above predates the flip - owner ruling
    // 2026-09-19 on the pin), the discriminator band on the kLoose direct's
    // own class, and the far field's liveness from the run's own record.
    // Second, the kNormal row CAN use the same-preset direct as its reference
    // (sound at this preset: the K halves are identical code at an equal
    // preset, and the CSV's C24-kNormal J-half error is 1.9e-9, over an order
    // under the 1e-7 budget).
    //
    // WHY THIS FIXTURE HAS NO SMALLER STAND-IN, measured on this tree
    // 2026-09-18 (owner fixture-size rule). A far-field row is only a test
    // of the far field where the far field MOVES THE ANSWER: the row's
    // sensitivity is the far field's own contribution to the converged
    // energy, |E_composed - E_composed(theta = -1)|, the gate run being the
    // same fixture, preset and SCF gate with the multipoles switched off.
    // Measured: C6 9.0e-11 (20 far pairs at kNormal), C8 2.3e-10 (206 far
    // pairs) - three orders BELOW the 1e-7 band the kNormal row asserts, so
    // a small fixture reaches the rung in PAIR COUNT and not in EFFECT, and
    // the row would pass there with the multipole machinery broken (a green
    // and useless pin, which is worse than a red one). The far field's
    // effect grows with the molecule, which is why the kNormal row is
    // asserted here and not at C6/C8. C24's far pairs at the corrected
    // The far-field default and the far field's effect on THIS fixture are printed
    // by the runs below.
    //
    // THE kTight ROW IS NOT HERE: the degenerate rung's claim is
    // size-independent by mechanism (the far field is empty at that preset)
    // and is asserted at C6 in
    // C6ChainQfmmRhfMatchesDirectInsideThePresetBudgets.
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto molecule = qcx::testing::MakeAlkaneSto3g(24);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // The fixture's basis-function total, counted through the framework's
    // own parser - the same basis text the runs below parse, so the count
    // is the runs' n_basis. The size-class band rows print it (the
    // report's bin assignment is machine-counted, never arithmetic).
    const auto chainBasis = qcx::testing::MakeAlkaneSto3gBasis();
    ASSERT_TRUE(chainBasis.has_value()) << chainBasis.error().message;
    const int basisCount =
        static_cast<int>(qcx::driver::CountBasisFunctions(*molecule, *chainBasis));

    // The measured composed rows of this test, accumulated for the
    // aggregate size-class band table printed at the end .
    std::vector<qcx::driver::QfmmAccuracyRunRow> bandRows;

    // The direct runs at all three presets, shared by the rows below.
    const auto ktightDirectToml = AlkaneChainToml(*molecule, "direct", "kTight");
    auto ktightResult = RunInputText(ktightDirectToml.c_str());
    ASSERT_TRUE(ktightResult.has_value()) << ktightResult.error().message;
    EXPECT_NE(ktightResult->find("\"converged\": true"), std::string::npos);
    const auto ktightDirectTotal = JsonNumber(*ktightResult, "\"total_energy_hartree\"");
    ASSERT_TRUE(ktightDirectTotal.has_value());

    const auto klooseDirectToml = AlkaneChainToml(*molecule, "direct", "kLoose");
    auto klooseResult = RunInputText(klooseDirectToml.c_str());
    ASSERT_TRUE(klooseResult.has_value()) << klooseResult.error().message;
    EXPECT_NE(klooseResult->find("\"converged\": true"), std::string::npos);
    const auto klooseDirectTotal = JsonNumber(*klooseResult, "\"total_energy_hartree\"");
    ASSERT_TRUE(klooseDirectTotal.has_value());

    const auto knormalDirectToml = AlkaneChainToml(*molecule, "direct", "kNormal");
    auto knormalResult = RunInputText(knormalDirectToml.c_str());
    ASSERT_TRUE(knormalResult.has_value()) << knormalResult.error().message;
    EXPECT_NE(knormalResult->find("\"converged\": true"), std::string::npos);
    const auto knormalDirectTotal = JsonNumber(*knormalResult, "\"total_energy_hartree\"");
    ASSERT_TRUE(knormalDirectTotal.has_value());

    // kLoose rows: the composed run vs near-exact at the C24
    // exchange-class budget 2e-4 (measured 2026-09-03: composed
    // 1.29971e-4, kLoose direct 1.49943e-4 - the fixture's own class, not
    // C12's) and the discriminator band on the kLoose direct's own class
    // (keeps the budget's basis live-checked on every gate run, as on
    // C12). The asymmetry between the two deviations is REPORTED, not
    // asserted - the note at the band below carries the measurement - and
    // the far field behind it is the one the rung assertion below holds live
    // (the run's own count: 73785 pairs, measured 2026-09-20; the accuracy record
    // record's 1997 is the same claim under the retired midpoint-extent
    // model, kept alive at the builder level by
    // QfmmHfBuildTest.C24KeepsItsFarAliveRungsAliveAtCreateTime).
    const auto qfmmKlooseToml = AlkaneChainToml(*molecule, "qfmm", "kLoose");
    auto qfmmKlooseResult = RunInputText(qfmmKlooseToml.c_str());
    ASSERT_TRUE(qfmmKlooseResult.has_value()) << qfmmKlooseResult.error().message;
    EXPECT_NE(qfmmKlooseResult->find("\"converged\": true"), std::string::npos);
    const auto qfmmKlooseTotal = JsonNumber(*qfmmKlooseResult, "\"total_energy_hartree\"");
    ASSERT_TRUE(qfmmKlooseTotal.has_value());

    // The kLoose row's rung, from the run's own record: the far field must be
    // LIVE. The run states that as its own geometry - the kLoose rung's angle
    // (0.45) and the far-pair count it admitted (73785 on this fixture,
    // measured 2026-09-20) - and a count of 0 here is the degenerate gate the
    // kTight row runs, whatever word the record carries. The discriminated
    // rows below are read as a statement about the composed path's far field,
    // so a degenerate run must fail this rather than quietly pass them.
    EXPECT_NE(qfmmKlooseResult->find("\"theta\": 0.45"), std::string::npos)
        << "the kLoose rung's own angle";
    EXPECT_NE(qfmmKlooseResult->find("\"geometric_far_pair_count\": 73785,"), std::string::npos)
        << "the kLoose composed run must record a live far field";

    const double directLooseDeviation = std::abs(*klooseDirectTotal - *ktightDirectTotal);
    const double klooseNearExactDeviation = std::abs(*qfmmKlooseTotal - *ktightDirectTotal);
    std::cout << "C24 preset kLoose: E_qfmm " << *qfmmKlooseTotal << ", E_direct-kTight "
              << *ktightDirectTotal << ", difference " << klooseNearExactDeviation << "\n";
    std::cout << "C24 kLoose direct vs kTight direct: " << directLooseDeviation << "\n";
    EXPECT_GT(directLooseDeviation, 1e-5) << "the kLoose direct's own screening class";
    EXPECT_LT(directLooseDeviation, 1e-3) << "the kLoose direct's own screening class";
    EXPECT_LT(klooseNearExactDeviation, 9.1e-4) << "preset kLoose";

    // This cell asserts the band above and NOT that the composed side beats the
    // kLoose direct. That relationship was measured and the loss accepted
    // (2026-09-19): composed 6.074926e-4 against the direct's 1.493365e-4, so
    // the composed deviation is 4.07x the direct's - and the two measure
    // different quantities, the direct's own screening class against the
    // composed path's distance from the kTight direct total, which the composed
    // route never promised to shrink. The band is this cell's acceptance; the
    // relationship is reported here rather than checked.

    // The size-class band line of this composed row (additive
    // REPORTING - the per-run band is text-only, never a budget, and
    // never enters the result JSON). At C24 this run's context does not
    // isolate the halves: the direct class above conflates the direct's
    // own screened-J far-region drop with the exchange half's class (the
    // two deviations' relationship was measured 2026-09-19 to invert -
    // composed 6.074926e-4 against the direct's 1.493365e-4, reported at
    // the band above and not asserted there), and the CSV J-half record
    // is not an in-run measurement - so the line prints the bin and the
    // composed deviation only.
    {
        qcx::driver::QfmmAccuracyRunRow bandRow;
        bandRow.fixture = "C24H50/STO-3G";
        bandRow.basisCount = basisCount;
        bandRow.preset = qcx::integrals::AccuracyPreset::kLoose;
        bandRow.reference = "kTight direct";
        bandRow.composedDeviation = klooseNearExactDeviation;
        std::cout << qcx::driver::FormatSizeClassBandLine(bandRow) << "\n";
        bandRows.push_back(bandRow);
    }

    // kNormal row: the sound same-preset comparison at the recorded 1e-7
    // budget (identical K halves; the J half is the CSV's 1.9e-9).
    const auto qfmmKnormalToml = AlkaneChainToml(*molecule, "qfmm", "kNormal");
    auto qfmmKnormalResult = RunInputText(qfmmKnormalToml.c_str());
    ASSERT_TRUE(qfmmKnormalResult.has_value()) << qfmmKnormalResult.error().message;
    EXPECT_NE(qfmmKnormalResult->find("\"converged\": true"), std::string::npos);
    const auto qfmmKnormalTotal = JsonNumber(*qfmmKnormalResult, "\"total_energy_hartree\"");
    ASSERT_TRUE(qfmmKnormalTotal.has_value());

    // The rung is asserted from the run's OWN record - the far field must be
    // LIVE in the run that produced the number below. The record states that
    // as the kNormal rung's own angle (0.3) and the far-pair count it admitted
    // (33404 on this fixture, measured 2026-09-20): a run that degenerated
    // here would record 0 and would compare two near-field builds, proving
    // nothing about the far field. The differential evidence that this
    // fixture is the smallest one where that live far field MOVES the answer
    // - the far field's own contribution to the converged energy - is in this
    // test's header note.
    EXPECT_NE(qfmmKnormalResult->find("\"theta\": 0.3"), std::string::npos)
        << "the kNormal rung's own angle";
    EXPECT_NE(qfmmKnormalResult->find("\"geometric_far_pair_count\": 33404,"), std::string::npos)
        << "the kNormal composed run must record a live far field";

    const double knormalDifference = std::abs(*qfmmKnormalTotal - *knormalDirectTotal);
    std::cout << "C24 preset kNormal: E_qfmm " << *qfmmKnormalTotal << ", E_direct-kNormal "
              << *knormalDirectTotal << ", difference " << knormalDifference << "\n";
    EXPECT_LT(knormalDifference, 1e-7) << "preset kNormal";

    // The kNormal composed row's band line: the same-preset context
    // measures the composed total only (the CSV J-half record is not an
    // in-run measurement), so the line prints the bin and the composed
    // deviation only.
    {
        qcx::driver::QfmmAccuracyRunRow bandRow;
        bandRow.fixture = "C24H50/STO-3G";
        bandRow.basisCount = basisCount;
        bandRow.preset = qcx::integrals::AccuracyPreset::kNormal;
        bandRow.reference = "kNormal direct";
        bandRow.composedDeviation = knormalDifference;
        std::cout << qcx::driver::FormatSizeClassBandLine(bandRow) << "\n";
        bandRows.push_back(bandRow);
    }

    // kTight row: MOVED to C6ChainQfmmRhfMatchesDirectInsideThePresetBudgets
    // (owner fixture-size rule, 2026-09-18 - see this test's header note).
    // The claim and its 1e-9 band are unchanged there, at ~65 s instead of
    // this fixture's composed-kTight run. The near-exact kTight DIRECT run
    // stays above: it is the kLoose rows' reference.

    // The aggregate size-class band table over this test's measured rows
    // (the pre-registered-bin schema: n_fixtures, N_basis range,
    // max/median/p95 |dE|, worst fixture; each (preset, reference) group
    // here holds one fixture, so the n < 3 underpopulation marker fires
    // honestly and no fit is claimed).
    std::cout << qcx::driver::FormatSizeClassBandTable(bandRows) << "\n";
}

#if !defined(QcxHasCuda)
TEST(DriverErrorTest, GpuWithoutCudaFallsBackToTheHeuristicPick) {
    // The device-less fallback: an explicit "gpu" without a CUDA device no longer refuses the
    // run - it falls back to the heuristic pick (direct at H2) with a loud
    // warning in the record and on stderr. The GPU builder's implementation
    // still lives in the CUDA TUs, which the default build never compiles;
    // the fallback is what the no-device lanes wire (the CUDA side of the
    // pair is selection_reporting_test.cpp's CudaGpuRequestWiresGpu...).
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const auto gpuWithoutCuda = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "gpu"
accuracy = "kNormal"
)");
    ASSERT_TRUE(gpuWithoutCuda.has_value()) << gpuWithoutCuda.error().message;

    // The fallback record: the pick wired, the explicit request flagged,
    // and the loud warning naming the missing device.
    const auto selection = gpuWithoutCuda->find("\"selection\":");
    ASSERT_NE(selection, std::string::npos);
    EXPECT_NE(gpuWithoutCuda->find("\"builder\": \"direct\"", selection), std::string::npos);
    EXPECT_NE(gpuWithoutCuda->find("\"explicit_builder\": \"gpu\"", selection), std::string::npos);
    EXPECT_NE(gpuWithoutCuda->find("no CUDA device", selection), std::string::npos);
}
#endif

TEST(DriverErrorTest, RejectsUnknownElementsAndResolvesTheAuxDefault) {
    const auto unknownElement = RunInputText(R"(
[molecule]
atoms = [["Xx", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_FALSE(unknownElement.has_value());
    EXPECT_EQ(unknownElement.error().code, qcx::ErrorCode::kInvalidArgument);

    // Re-pinned 2026-09-14, and the test is RENAMED with it: this leg used to
    // assert that a sto-3g orbital basis with fock_builder = "ri_j_link" is
    // REFUSED, on the reading that SelectAuxBasis has no rule for sto-3g
    //  so the driver said so "instead of silently pairing a wrong fit".
    // That refusal is gone BY DESIGN - the aux-auto rule (5b21a295, 2026-09-13 21:44,
    // owner-relayed ruling) made the aux rule a quality tier that ALWAYS
    // resolves a default: a dedicated matched fit when one exists, otherwise
    // the universal fit. So the request now resolves and the run proceeds.
    //
    // Note the attribution, because it is not this file's ladder: The aux-auto rule is
    // its own landed change, and the default-family ladder (afca4a2e) does
    // not touch the aux rule at all.
    //
    // What is pinned: the run is ACCEPTED and converges, and the default the
    // rule picks is a fit that reproduces the direct route's own pin to the
    // RI-J error class (measured 2026-09-14: -1.1168279600 against the
    // direct -1.1167143252, i.e. 1.14e-4 BELOW it - the variational
    // signature of a working fit, the RI-J tolerance here being the repo's
    // standing 1e-3). Pinning the FIT rather than only the acceptance is
    // what keeps this leg from degenerating into a rubber stamp: "no longer
    // refused" alone would pass for a silently wrong pairing too.
    //
    // PINNED SINCE THE NOTICE'S PRODUCER LANDED (2026-09-16). This leg used to
    // say the NAME of the resolved aux was NOT pinned and could not be:
    // the record plumbing for that (AuxSelectionNotice) was open and unlanded,
    // and the warnings the aux-auto rule calls for were structurally blocked,
    // so nothing on this run's face said which fit was chosen. That sentence
    // was true when it was written and is false now,
    // so the assertion below carries the fact instead. The energy pin is NOT
    // weakened or moved by it: the two are independent, and the fit's name does
    // not make the number right.
    //
    // The region is why this run owes a notice at all: sto-3g is a MINIMAL
    // basis, the strongest residual in the measurement the ruling rests
    // on, and the notice's own record home is the approximation block whose
    // `aux_basis` is read here (`TheAuxWeakRegionNoticeReachesTheRunRecord`
    // pins the notice text and the presence rule).
    const auto missingAux = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
)");
    ASSERT_TRUE(missingAux.has_value())
        << "the aux rule always resolves a default; a refusal here means the "
           "quality tier is gone again: "
        << missingAux.error().message;
    EXPECT_NE(missingAux->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(missingAux->find("\"aux_basis\": \"def2-universal-jfit\""), std::string::npos)
        << *missingAux;

    const auto total = JsonNumber(*missingAux, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, -1.1167143252, 1e-3);
}

TEST(DriverErrorTest, FailsFastOnSemanticallyInvalidInput) {
    // Multiplicity 2 with 2 electrons violates the parity rule; the driver
    // must reject it up front (kInvalidArgument, first validation issue)
    // instead of silently running the wrong spin state.
    const auto parityViolation = RunInputText(R"(
[molecule]
multiplicity = 2
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74084809526419992, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_FALSE(parityViolation.has_value());
    EXPECT_EQ(parityViolation.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(parityViolation.error().message.find("multiplicity"), std::string::npos);
}

TEST(DriverErrorTest, RejectsRhfWithNonSingletMultiplicity) {
    // The O2 triplet under method rhf: 16 electrons and multiplicity 3 pass
    // the parity rule, but the RHF lane is closed-shell by construction -
    // before the fix the run silently reported the singlet state as correct
    // (spin_squared is null for RHF, so nothing in-band flagged it). The
    // driver must reject the request up front like the other semantic
    // violations.
    const auto tripletRhf = RunInputText(R"(
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
type = "rhf"
fock_builder = "direct"
accuracy = "kTight"
)");
    ASSERT_FALSE(tripletRhf.has_value());
    EXPECT_EQ(tripletRhf.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(tripletRhf.error().message.find("multiplicity"), std::string::npos);
}

TEST(DriverRunTest, RiJLinkAutoSelectsTheAuxBasis) {
    if (qcx::testing::IsFastOnlyMode() || !qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "heavy numerical test: Release-only; the universal-J aux f "
                        "shells exceed this build's kMaxEngineL (CI lmax=2)";
    }

    // def2-svp has a bundled universal-J aux; omitting [basis].aux
    // exercises the auto-selection through the driver path. No pin exists
    // for the RI path here - the run must just converge and serialize.
    auto result = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "def2-svp"

[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kLoose"
)");
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

    EXPECT_NE(result->find("\"builder\": \"ri_j_link\""), std::string::npos);

    // The budget path (the budget path): the default 16.0 GiB cap becomes
    // a workspace budget of the cap MINUS the run's reserve - the engine's
    // own ramp-and-sweeps bound (SetupRampPeakBytes) times its one slack
    // constant (kSetupAdmissionSlack, 1.5), computed ONCE by the pre-gate
    // setup arm and threaded to every budget site
    // (integrals::SetupAdmissionReserveBytes). H2O/def2-svp with the
    // auto-selected universal-J aux in effect: the bound is 10,036,560 B, so
    // the reserve is the 15,054,840 B above it and the capacity is
    // 16.0 * 2^30 - 15,054,840 = 17164814344 bytes.
    EXPECT_NE(result->find("\"workspace_budget\""), std::string::npos);
    EXPECT_NE(result->find("\"capacity_bytes\": 17164814344"), std::string::npos);
    EXPECT_NE(result->find("\"committed_bytes\""), std::string::npos);

    // The mode record: H2O/def2-SVP is tiny, so the Create-time estimate
    // admits the fast path with no exclusions. The mode value keeps the
    // engine's enumerator vocabulary (the kFastPath pin).
    EXPECT_NE(result->find("\"mode_record\""), std::string::npos);
    EXPECT_NE(result->find("\"mode\": \"kFastPath\""), std::string::npos);
    EXPECT_NE(result->find("\"pattern_excluded\": false"), std::string::npos);
    EXPECT_NE(result->find("\"tensor_excluded\": false"), std::string::npos);

    // The full-group labeling block: H2O is C2v by construction
    // (the mirror-plane geometry above), so the labels run by default.
    EXPECT_NE(result->find("\"symmetry\""), std::string::npos);
    EXPECT_NE(result->find("\"full_group\": \"C2v\""), std::string::npos);
    EXPECT_NE(result->find("\"abelian_reduction\": \"C2v\""), std::string::npos);
    EXPECT_NE(result->find("\"labels\""), std::string::npos);
    // symmetry_beta is the UHF-only member: absent on this RHF run.
    EXPECT_EQ(result->find("\"symmetry_beta\""), std::string::npos);

    const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_LT(*total, 0.0);
}

// The symmetry-blocking disclosure at the driver level: the
// run record's own account of the guard's decision, per spin channel. The
// fixture is a measured probe: stretched H2/STO-3G from
// the core guess, the units key explicit so the 3.5 is read as bohr, and no
// [scf] block - the operating defaults. Both spin Focks stay adapted to the
// abelian reduction's FOUR generators, so every diagonalization ran blocked
// and none was refused.
//
// The word and the evidence beside it are pinned as the pair they are: the
// counts and the norms are what a reader CHECKS the decision against, and
// `kUsed` is what the deciding code said. The magnitude is asserted for what
// it IS - round-off, orders below the threshold - and not against a recorded
// value: a commutator norm at the noise floor is a property of the arithmetic,
// while the last digits of one machine's round-off are a property of that
// machine. The refused sibling below pins the other word on the other
// evidence - these two cells exist because a record that wrote one word for
// both kinds of run must not pass unnoticed.
TEST(DriverRunTest, SymmetryBlockingReadsUsedWhenTheGuardBlocked) {
    auto result = RunInputText(R"(
[molecule]
units = "bohr"
charge = 0
multiplicity = 1
atoms = [
    ["H", 0.0, 0.0, 0.0],
    ["H", 3.5, 0.0, 0.0],
]

[basis]
orbital = "sto-3g"

[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"

[guess]
type = "core"
)");
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

    const auto blockPos = result->find("\"symmetry_blocking\"");
    ASSERT_NE(blockPos, std::string::npos) << "the guard ran, so the record must account for it";

    // The block's two channels are separate objects, alpha first (the
    // serializer's key order). Everything below is asserted per channel, on
    // the region that channel owns: the alpha and beta Focks are different
    // matrices, so the guard's verdict is per-spin.
    const auto betaPos = result->find("\"beta\":", blockPos);
    ASSERT_NE(betaPos, std::string::npos) << "the beta channel object";

    const auto alphaAction = result->find("\"action\": \"kUsed\"", blockPos);
    ASSERT_NE(alphaAction, std::string::npos) << "alpha";
    EXPECT_LT(alphaAction, betaPos) << "the block's first word is the alpha channel's";
    EXPECT_NE(result->find("\"action\": \"kUsed\"", betaPos), std::string::npos) << "beta";
    // Neither channel refused and neither walked both paths.
    EXPECT_EQ(result->find("\"action\": \"kRefused\"", blockPos), std::string::npos);
    EXPECT_EQ(result->find("\"action\": \"kDemoted\"", blockPos), std::string::npos);

    for (const auto channelPos : {blockPos, betaPos})
    {
        const auto blocked = JsonScalarNumber(*result, "\"blocked_solve_count\":", channelPos);
        const auto plain = JsonScalarNumber(*result, "\"plain_solve_count\":", channelPos);
        ASSERT_TRUE(blocked.has_value());
        ASSERT_TRUE(plain.has_value());
        EXPECT_EQ(static_cast<int>(*blocked), 2) << "the count the kUsed word was decided on";
        EXPECT_EQ(static_cast<int>(*plain), 0);

        const auto tolerance = JsonScalarNumber(*result, "\"tolerance\":", channelPos);
        const auto maximum =
            JsonScalarNumber(*result, "\"max_generator_commutator_norm\":", channelPos);
        ASSERT_TRUE(tolerance.has_value());
        ASSERT_TRUE(maximum.has_value());
        EXPECT_DOUBLE_EQ(*tolerance, 1e-08);
        // The magnitude is the ARITHMETIC'S noise floor, so the assertion is
        // the noise floor and not the reading. The guard's decision is the
        // comparison `maximum < tolerance`, and the margin below is what makes
        // that comparison decisive rather than marginal: a quarter of the
        // threshold is not a statement about an exactly commuting Fock, and
        // equality is not either, but a commutator that sits four orders below
        // the line can only be round-off. Four orders is also about four
        // thousand machine epsilons of a unit-scale Fock - wider than any
        // accumulation difference between the platforms this builds on, and
        // narrower than any breaking that is real: the refused sibling's
        // symmetry violation lands at 1e-2, ten orders above this bound.
        //
        // Evidence, not contract (2026-09-19): the reading is 9.1841325126e-16
        // on the two arm64 legs and 2.9644138242e-15 on windows-msvc, i.e.
        // 1.3e-8 to 4.2e-8 OF the tolerance - the four orders asserted here
        // with a factor of thirty in hand.
        EXPECT_LT(*maximum, *tolerance * 1e-4) << "the magnitude the kUsed word was decided on";
        EXPECT_LT(*maximum, *tolerance) << "the guard's own comparison";

        // The recorded norms are the WORST measurement's (scf/uhf.hpp), one
        // per generator of the computational group, so they always explain the
        // recorded maximum - the number a reader verifies the decision
        // against rather than takes.
        const auto norms = JsonNumberArray(*result, "\"generator_commutator_norms\":", channelPos);
        ASSERT_TRUE(norms.has_value());
        EXPECT_EQ(norms->values.size(), 4U);

        double worst = 0.0;

        for (const double norm : norms->values)
        {
            worst = std::max(worst, norm);
        }

        EXPECT_DOUBLE_EQ(worst, *maximum);
    }
}

// The other half of the disclosure's vocabulary, on the run it was measured
// it on: O2/STO-3G read as a SINGLET (multiplicity 1, so both spins start from
// one closed-shell-like guess) from the core guess at R = 2.2816 bohr
// (build/windows-msvc/probe/o2s-core.toml, verbatim). The alpha and beta Focks
// break the group's symmetry from the FIRST Fock on, so the guard refused
// every diagonalization - and the two generators the breaking reaches stand
// orders of magnitude above the 1e-8 threshold, which is why.
//
// This is the arm a reader cannot infer from the run: a refused blocking still
// converges to a variational-looking answer, so without this block the record
// would read as though the irrep blocks had been used.
TEST(DriverRunTest, SymmetryBlockingReadsRefusedWhenTheGuardNeverBlocked) {
    auto result = RunInputText(R"(
[molecule]
units = "bohr"
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 2.2816],
]

[basis]
orbital = "sto-3g"

[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"

[guess]
type = "core"
)");
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

    const auto blockPos = result->find("\"symmetry_blocking\"");
    ASSERT_NE(blockPos, std::string::npos) << "blocking was requested, so the record must say so";
    const auto betaPos = result->find("\"beta\":", blockPos);
    ASSERT_NE(betaPos, std::string::npos) << "the beta channel object";

    const auto alphaAction = result->find("\"action\": \"kRefused\"", blockPos);
    ASSERT_NE(alphaAction, std::string::npos) << "alpha";
    EXPECT_LT(alphaAction, betaPos) << "the block's first word is the alpha channel's";
    EXPECT_NE(result->find("\"action\": \"kRefused\"", betaPos), std::string::npos) << "beta";
    EXPECT_EQ(result->find("\"action\": \"kUsed\"", blockPos), std::string::npos);
    EXPECT_EQ(result->find("\"action\": \"kDemoted\"", blockPos), std::string::npos);

    for (const auto channelPos : {blockPos, betaPos})
    {
        const auto blocked = JsonScalarNumber(*result, "\"blocked_solve_count\":", channelPos);
        const auto plain = JsonScalarNumber(*result, "\"plain_solve_count\":", channelPos);
        ASSERT_TRUE(blocked.has_value());
        ASSERT_TRUE(plain.has_value());
        EXPECT_EQ(static_cast<int>(*blocked), 0) << "the count the kRefused word was decided on";
        EXPECT_EQ(static_cast<int>(*plain), 10);

        const auto tolerance = JsonScalarNumber(*result, "\"tolerance\":", channelPos);
        const auto maximum =
            JsonScalarNumber(*result, "\"max_generator_commutator_norm\":", channelPos);
        ASSERT_TRUE(tolerance.has_value());
        ASSERT_TRUE(maximum.has_value());
        EXPECT_DOUBLE_EQ(*tolerance, 1e-08);
        // The refusal is decided by a magnitude that is orders above the line,
        // so that is what the assertion states - three orders, because one
        // order is indistinguishable from a marginal float and the guard's
        // question is whether the symmetry broke at all. The magnitude itself
        // is a property of the TRAJECTORY (how far this run's Fock departs
        // from the reduction), so it moves with the machine and is not the
        // contract; that it stands clear of the threshold is.
        //
        // Evidence, not contract (2026-09-19): the reading is 5.3704369502e-03
        // on the linux-arm64 and macos arm64 legs and 2.4264535832e-02 on
        // windows-arm64 - 5.4e5 and 2.4e6 times the tolerance, i.e. 500x and
        // 2400x above the bound asserted here, the same direction on every leg
        // and a factor of 4.5 spread between them.
        EXPECT_GT(*maximum, *tolerance * 1e3) << "the magnitude the kRefused word was decided on";
        EXPECT_GT(*maximum, *tolerance) << "the guard's own comparison";

        const auto norms = JsonNumberArray(*result, "\"generator_commutator_norms\":", channelPos);
        ASSERT_TRUE(norms.has_value());
        EXPECT_EQ(norms->values.size(), 4U);

        double worst = 0.0;

        for (const double norm : norms->values)
        {
            worst = std::max(worst, norm);
        }

        EXPECT_DOUBLE_EQ(worst, *maximum);
        // TWO of the four generators are the broken ones; the other two sit at
        // round-off, so the refusal is about a direction that really breaks.
        EXPECT_EQ(
            std::count_if(norms->values.begin(),
                          norms->values.end(),
                          [tolerance = *tolerance](double value) { return value > tolerance; }),
            2);
    }
}

// The third word of the vocabulary, on a fixture the ordinary input surface
// reaches it through: water with one H lifted 0.15 Angstrom along z. The three
// nuclei stay coplanar, so the detected group is genuinely Cs - a three-atom
// molecule is ALWAYS at least Cs, since three points lie in a plane and that
// plane is a mirror operation on the nuclei - while the unrestricted solution
// that converges is not adapted to it. The run therefore walks BOTH paths:
// the early diagonalizations block and the later ones are refused, one word
// for the run. Measured on this fixture: 3 blocked / 7 plain at the 0.15
// Angstrom lift, and the same word at 0.05, 0.3 and 0.5 (4/7, 3/8, 4/8) - the
// split moves with the trajectory, which is why this cell pins the WORD and
// the direction of both counts rather than the split.
//
// It is the word the schema doc calls out as the one a reader must tell apart
// from kUsed, and the only one whose evidence is mixed by design: a non-zero
// blocked count beside a non-zero plain one.
TEST(DriverRunTest, SymmetryBlockingReadsDemotedWhenTheRunWalkedBothPaths) {
    auto result = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.15],
]

[basis]
orbital = "sto-3g"

[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"

[guess]
type = "core"
)");
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

    const auto blockPos = result->find("\"symmetry_blocking\"");
    ASSERT_NE(blockPos, std::string::npos) << "the guard ran, so the record must account for it";
    const auto betaPos = result->find("\"beta\":", blockPos);
    ASSERT_NE(betaPos, std::string::npos) << "the beta channel object";

    const auto alphaAction = result->find("\"action\": \"kDemoted\"", blockPos);
    ASSERT_NE(alphaAction, std::string::npos) << "alpha";
    EXPECT_LT(alphaAction, betaPos) << "the block's first word is the alpha channel's";
    EXPECT_NE(result->find("\"action\": \"kDemoted\"", betaPos), std::string::npos) << "beta";
    // Neither all-blocked (kUsed) nor all-plain (kRefused): the run walked both.
    EXPECT_EQ(result->find("\"action\": \"kUsed\"", blockPos), std::string::npos);
    EXPECT_EQ(result->find("\"action\": \"kRefused\"", blockPos), std::string::npos);

    for (const auto channelPos : {blockPos, betaPos})
    {
        const auto blocked = JsonScalarNumber(*result, "\"blocked_solve_count\":", channelPos);
        const auto plain = JsonScalarNumber(*result, "\"plain_solve_count\":", channelPos);
        ASSERT_TRUE(blocked.has_value());
        ASSERT_TRUE(plain.has_value());
        EXPECT_GT(static_cast<int>(*blocked), 0) << "the blocked path was walked at least once";
        EXPECT_GT(static_cast<int>(*plain), 0) << "the plain path was walked at least once";

        const auto tolerance = JsonScalarNumber(*result, "\"tolerance\":", channelPos);
        const auto maximum =
            JsonScalarNumber(*result, "\"max_generator_commutator_norm\":", channelPos);
        ASSERT_TRUE(tolerance.has_value());
        ASSERT_TRUE(maximum.has_value());
        EXPECT_DOUBLE_EQ(*tolerance, 1e-08);
        EXPECT_GT(*maximum, *tolerance) << "the magnitude the refusals were decided on";

        const auto norms = JsonNumberArray(*result, "\"generator_commutator_norms\":", channelPos);
        ASSERT_TRUE(norms.has_value());
        EXPECT_EQ(norms->values.size(), 1U) << "the Cs reduction has one generator";

        double worst = 0.0;

        for (const double norm : norms->values)
        {
            worst = std::max(worst, norm);
        }

        EXPECT_DOUBLE_EQ(worst, *maximum);
    }
}

// The presence rule's other arm: a run that never asked carries NO
// block - not a null block, not a fabricated `kUsed`, and not the fabricated
// `kNotRequested` word either, because the guard's enum value for "never
// asked" IS the key's absence (kNotRequested is the absence, never a word in
// the record). The fixture is distorted hydrogen peroxide: the two O-H bonds
// differ in length and dihedral, so the molecule has NO symmetry element and
// the detected group is C1 - unlike a three-atom fixture, which is always at
// least Cs (see the demoted cell above). Nothing is requested, so no
// decomposition is built and the guard has nothing to decide.
TEST(DriverRunTest, SymmetryBlockingAbsentWhenNothingWasRequested) {
    auto result = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["O", 0.0, 0.0, 1.4],
    ["H", 0.9, 0.4, -0.3],
    ["H", -0.5, 0.9, 1.6],
]

[basis]
orbital = "sto-3g"

[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"

[guess]
type = "core"
)");
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

    // The control: the UHF tail that writes the block wrote this run's
    // spin_squared (never null on a UHF run), so the absence below is this
    // run's own decision and not a code path that never ran.
    EXPECT_NE(result->find("\"spin_squared\":"), std::string::npos);
    EXPECT_EQ(result->find("\"spin_squared\": null"), std::string::npos);

    // Searching for the KEY is what tells absence from a fabricated null
    // block, which would carry the key with no honest statement inside it.
    EXPECT_EQ(result->find("\"symmetry_blocking\""), std::string::npos);
    // No action word anywhere either: `action` is this block's member and no
    // other block's, so a fabricated word for a run that never asked has
    // nowhere to hide.
    EXPECT_EQ(result->find("\"action\""), std::string::npos);

    // The labeling blocks agree, one block over: the trivial group carries no
    // information, so the full-group blocks are absent for the same C1 cause.
    EXPECT_EQ(result->find("\"symmetry\""), std::string::npos);
    EXPECT_EQ(result->find("\"symmetry_beta\""), std::string::npos);
}

// The disk-rung knob (
// 6.6) through the error surface: the io layer's vocabulary rejections run
// through the full input pipeline, and the `disk` word refuses loudly on
// the legacy null-budget path - the memory_cap_gib = 0 escape hatch keeps
// a budget-less engine ladder that never refuses, so the knob would be a
// silent no-op there (for that word the disk rung can only engage on the
// in-memory ladder's estimate-time refusal, and a refusal needs a budget;
// a cap between zero and the base term already refused above with the
// standing reinstatement message). The FORCE is a separate key in its own
// table ([diagnostics] force_disk_ri, the owner's ruling 2026-09-13) and
// has no ladder to refuse from, so it engages the disk builder with no
// budget granted - the bypass the forced-key test below measures against a
// fitting ladder.
TEST(DriverErrorTest, RiTensorSchemaKnobsRejectedThroughThePipeline) {
    const auto badMode = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
ri_tensor_mode = "ram"
)");
    ASSERT_FALSE(badMode.has_value());
    EXPECT_EQ(badMode.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(badMode.error().message.find("method.ri_tensor_mode"), std::string::npos);
    EXPECT_NE(badMode.error().message.find("(auto | disk)"), std::string::npos);

    // The retired force word, through the same pipeline: it is refused with
    // the key that REPLACED it named, because the capability moved rather
    // than disappeared. The assertion is on the remedy, not only on the
    // refusal: a reader whose file stopped parsing must be told where the
    // force went, and an error listing only the two surviving rung words
    // would send them hunting for a feature that is still there.
    const auto retiredWord = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
ri_tensor_mode = "forced_disk"
)");
    ASSERT_FALSE(retiredWord.has_value());
    EXPECT_EQ(retiredWord.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(retiredWord.error().message.find("does not take \"forced_disk\""), std::string::npos);
    EXPECT_NE(retiredWord.error().message.find("[diagnostics] force_disk_ri = true"),
              std::string::npos);

    const auto badChunk = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
ri_chunk_bytes = 0
)");
    ASSERT_FALSE(badChunk.has_value());
    EXPECT_EQ(badChunk.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(badChunk.error().message.find("method.ri_chunk_bytes"), std::string::npos);

    const auto legacyDisk = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
aux = "def2-universal-jfit"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
ri_tensor_mode = "disk"
[resources]
memory_cap_gib = 0
)");
    ASSERT_FALSE(legacyDisk.has_value());
    EXPECT_EQ(legacyDisk.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(legacyDisk.error().message.find("needs the budget path"), std::string::npos);

    // (The forced word on another family is no longer a refusal case: under
    // the enforcement contract it DEMOTES with the record stating both
    // sides - see RiTensorModeRequestsOnAnotherFamilyAreRecordedNotDropped
    // below. This test's cases are the ones that still refuse.)
}

// The other side of the key split (owner-ruled 2026-09-13):
// method.ri_chunk_bytes is a pure SIZE HINT - a chunk size for a tensor store
// the run either has or has not - so on a family with no disk rung it is NOT
// refused. The request is dropped and the run proceeds; refusing it there is
// the pedantic reading the ruling rejects, and the alternative it forbids is
// not refusal but silence about a key the run did use differently.
//
// Both halves of that ruling are landed. The DISCLOSURE is
// resources_resolved.ri_chunk_bytes: the size that was asked for,
// the outcome word and, on a drop, the reason - written whenever the input
// named the key, so the emitted document names the key the run dropped. That
// is the state this comment spent a day describing as OWED, and the paragraph
// here is rewritten rather than deleted because a reader who finds the
// proposal text quoted in an older commit should see where it landed. The
// block's full pin - the drop, the honour, and the ri_j_link ladder fit where
// the sibling block says ladder_fit about the rung request while this one says
// dropped - is driver/tests/ri_chunk_bytes_record_test.cpp; the two assertions
// below keep the disclosure visible in THIS row as well, since this is the row
// that pins the PROCEED half.
//
// What this row pins is therefore the PROCEED half - the half the ruling
// settles, and the leg that fails if the key is ever refused on a family with
// no disk rung - beside the record facts and the energy: the run completes,
// the record says the hint was dropped, and the energy is the direct family's.
TEST(DriverErrorTest, RiChunkBytesOnAnotherFamilyProceedsAndDropsTheKey) {
    const auto noDiskRung = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
ri_chunk_bytes = 134217728
)");
    ASSERT_TRUE(noDiskRung.has_value())
        << "the size hint was refused on a family with no chunking to size: "
        << noDiskRung.error().message;
    EXPECT_NE(noDiskRung->find("\"converged\": true"), std::string::npos);

    // The record half: the document names the key, what it asked
    // for, and what happened to it - the disclosure that closes the silence
    // this row used to describe as owed.
    EXPECT_NE(noDiskRung->find("\"ri_chunk_bytes\""), std::string::npos);
    EXPECT_NE(noDiskRung->find("\"outcome\": \"dropped\""), std::string::npos);
    EXPECT_NE(noDiskRung->find("\"requested_bytes\": 134217728"), std::string::npos);

    // The energy is the same one the identical input without the key reaches:
    // a chunk size that was never consumed cannot move a result, and the
    // run's record must not suggest the direct family chunked anything. The
    // comparison is taken in-process against the neighbouring run rather than
    // against a stored constant, so it cannot drift with the pins.
    const auto withoutKey = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_TRUE(withoutKey.has_value()) << withoutKey.error().message;

    // The third reader state, pinned here because it is the one a project-wide
    // search for the key must not confuse with a drop: a run that never named
    // it carries no block at all.
    EXPECT_EQ(withoutKey->find("ri_chunk_bytes"), std::string::npos);

    const auto knobbedEnergy = JsonNumber(*noDiskRung, "\"total_energy_hartree\"");
    const auto plainEnergy = JsonNumber(*withoutKey, "\"total_energy_hartree\"");
    ASSERT_TRUE(knobbedEnergy.has_value());
    ASSERT_TRUE(plainEnergy.has_value());
    EXPECT_DOUBLE_EQ(*knobbedEnergy, *plainEnergy);
}

// The orbit-expansion key on a family with no 3c task grid: the
// request is STATED, not dropped and not refused. The key's one consumer is
// the ri_j_link arm, so its request on any other family has nothing to act on
// - and the two failure modes this row exists to catch are a silent no-op
// (the request vanishes from the document) and an over-eager refusal (a key
// that is meaningless on this family stops a run that never needed it).
//
// The third reader state is pinned with them: a run that never named the key
// carries no block at all, so a project-wide search for `ri_orbit_expansion`
// cannot confuse "never asked" with "asked and not applicable".
TEST(DriverErrorTest, RiOrbitExpansionOnAnotherFamilyIsRecordedNotApplicable) {
    const auto noGrid = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
ri_orbit_expansion = true
)");
    ASSERT_TRUE(noGrid.has_value())
        << "the orbit-expansion key was refused on a family that evaluates no 3c grid: "
        << noGrid.error().message;
    EXPECT_NE(noGrid->find("\"converged\": true"), std::string::npos);

    // The record half: the document names the key, what it asked
    // for, and why nothing came of it on this family.
    EXPECT_NE(noGrid->find("\"ri_orbit_expansion\""), std::string::npos);
    EXPECT_NE(noGrid->find("\"requested\": true"), std::string::npos);
    EXPECT_NE(noGrid->find("\"outcome\": \"not_applicable\""), std::string::npos);
    EXPECT_NE(noGrid->find("does not consume the key"), std::string::npos);
    // A request that did nothing never claims it did.
    EXPECT_EQ(noGrid->find("\"outcome\": \"engaged\""), std::string::npos);

    const auto withoutKey = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_TRUE(withoutKey.has_value()) << withoutKey.error().message;
    EXPECT_EQ(withoutKey->find("ri_orbit_expansion"), std::string::npos);

    // A request that was never consumed cannot move a result: the same input
    // without the key reaches the same energy, compared in-process so the
    // assertion cannot drift with the pins.
    const auto knobbedEnergy = JsonNumber(*noGrid, "\"total_energy_hartree\"");
    const auto plainEnergy = JsonNumber(*withoutKey, "\"total_energy_hartree\"");
    ASSERT_TRUE(knobbedEnergy.has_value());
    ASSERT_TRUE(plainEnergy.has_value());
    EXPECT_DOUBLE_EQ(*knobbedEnergy, *plainEnergy);
}

// The requested-vs-ran pairing at the driver (demote-with-disclosure): a request
// the resolved family cannot honour no longer vanishes and is no longer
// refused - the run proceeds on that family and the record states what was
// asked for, what ran, and why they differ. This test REPLACES the interim
// by-name refusal this wiring landed earlier the same day: its assertions
// moved from "the run fails with a named refusal" to "the run completes and
// the record discloses the demotion" - a change of MEANING that only the
// comment makes visible, which is why the comment is here.
//
// This is a CHARACTERIZATION pin on today's disclosure contract, not a claim
// that the limitation is permanent: it fails if the record stops disclosing
// (the defect it exists to catch) AND if a family ever GAINS an RI disk rung,
// in which case the demotion becomes an honour and a human updates the pin
// deliberately rather than the record drifting unnoticed. It asserts a
// forward statement (what the record must say) and never a failure, so it
// cannot invert into punishing an improvement.
TEST(DriverRunTest, RiTensorModeRequestsOnAnotherFamilyAreRecordedNotDropped) {
    // An explicit family word that has no RI disk rung, with the FORCE named
    // (the key that replaced the retired `ri_tensor_mode = "forced_disk"`
    // word, the owner's ruling 2026-09-13): the request asserted which
    // algorithm ran, the run did not satisfy it, so it is demoted with the
    // family named in the reason and the block naming the KEY the request
    // came from. A demotion this record can state is the expected
    // outcome; the refusal posture the force key does NOT take is recorded
    // in run_driver.cpp's ValidateCombination note.
    const auto forced = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
[diagnostics]
force_disk_ri = true
)");
    ASSERT_TRUE(forced.has_value()) << forced.error().message;
    EXPECT_NE(forced->find("\"requested\": \"forced_disk\""), std::string::npos);
    EXPECT_NE(forced->find("\"resolved\": \"in_memory\""), std::string::npos);
    EXPECT_NE(forced->find("\"outcome\": \"demoted\""), std::string::npos);
    EXPECT_NE(forced->find("\"forced\": true"), std::string::npos);
    EXPECT_NE(forced->find("fock_builder = \\\"direct\\\" has no disk rung"), std::string::npos);

    // The absent family word spelling: the same demotion, and the record still
    // names the family the run resolved to - the honest fact, since the family
    // (not a word the file never wrote) is what could not honour it. This is
    // also the LEAN arm at this size, i.e. a path with no mode_record at
    // all: the pairing records the request where the mode record cannot.
    const auto absent = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
[diagnostics]
force_disk_ri = true
)");
    ASSERT_TRUE(absent.has_value()) << absent.error().message;
    EXPECT_NE(absent->find("\"resolved\": \"in_memory\""), std::string::npos);
    EXPECT_NE(absent->find("\"outcome\": \"demoted\""), std::string::npos);
    EXPECT_NE(absent->find("\"forced\": true"), std::string::npos);

    // The ladder word on the same family: NOT a demotion - `disk` permits
    // the ladder's last rung without demanding it, and a family with no rung
    // to offer simply had nothing to select. Recorded, never dropped, and
    // `forced` is false: the two request sources are told apart in the record
    // itself, not by which word the reader remembers writing.
    const auto ladderWord = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
ri_tensor_mode = "disk"
)");
    ASSERT_TRUE(ladderWord.has_value()) << ladderWord.error().message;
    EXPECT_NE(ladderWord->find("\"requested\": \"disk\""), std::string::npos);
    EXPECT_NE(ladderWord->find("\"resolved\": \"in_memory\""), std::string::npos);
    EXPECT_NE(ladderWord->find("\"outcome\": \"not_applicable\""), std::string::npos);
    EXPECT_NE(ladderWord->find("\"forced\": false"), std::string::npos);
    EXPECT_NE(ladderWord->find("nothing to select"), std::string::npos);
}

// The disk rung's driver wiring: method.ri_tensor_mode
// = "disk" engages the storage-module DiskRiFockBuilder as the ladder's
// LAST rung - only on the in-memory ladder's estimate-time refusal under a
// budget. Runs in the cap-child like the other small-cap tests (a small
// memory cap is a real job-object cap; a run that completes under it cannot
// live in the host test process). The fixture is H2O/STO-3G with the
// explicit universal-J aux (STO-3G has no auto-selection rule): n = 7,
// nAux = 71, so the disk record's modeled dense payload (the payload
// reconciliation target) is 8 * 49 * 71 = 27832 bytes. The 0.1 GiB cap
// comes from direct probes (2026-09-03): the driver's base-term
// refusal fires at cap <= 0.056 GiB (the n < 24 base floor), and the
// engine's Create-time estimate for THIS fixture at kTight read
// 7517325050 bytes (the light rung's charge), so the window in which the
// in-memory ladder refuses opened at 0.0561 GiB and 0.1 GiB sat inside it.
// That window is a reading of ONE host's plan, not a contract: on the
// linux-x86 gcc leg the same cap fits an in-memory rung (2026-09-19, CI run
// 35451596080), and the cell below follows the record's own outcome for that
// reason. The SCF must reproduce the storage-suite RI-J pin -74.96369193
// (disk_ri_fock_build_test.cpp), NOT the direct pin -74.96292827 - on
// whichever branch carries the disk route.
TEST(DriverRunTest, RiTensorModeDiskEngagesTheDiskRungAfterTheLaddersRefusal) {
#if !defined(QcxHasStorage)
    GTEST_SKIP() << "storage module not built (QCX_ENABLE_IO=OFF): the disk rung IS the storage "
                    "module's disk-backed builder, so there is no rung to engage";
#elif defined(QcxHasCuda) || defined(_DEBUG)
    // CUDA build only: same floor as the budget-refusal tests - the
    // runtime's load-time footprint (~0.25-0.3 GiB) exceeds any small
    // cap, so the CUDA-linked child dies before the seam . The Debug leg skips the same floor
    // (the Debug exe's load-time footprint also exceeds the tiny cap - the 2026-09-08 wedge class).
    // The disk
    // route is CPU-side; the CUDA-less builds carry this fixture.
    GTEST_SKIP()
        << "CUDA or Debug build: the runtime's load-time footprint (~0.25-0.3 GiB) exceeds "
           "the small cap; the disk-route contract is verified on the CUDA-less "
           "builds";
#else
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jfit g shells";
    }

    constexpr double kDiskCapGib = 0.1;
    const auto marker = qcx::testing::CapChildMarkerPath(13);
    std::filesystem::remove(marker);

    const int exitCode = qcx::testing::RunCapChild(kDiskCapGib, marker, "ri_j_disk");

    EXPECT_EQ(exitCode, 0)
        << "the disk-rung run must complete under the cap, not refuse or die at the cap";
    const std::string contents = qcx::testing::ReadCapChildMarker(marker);
    EXPECT_NE(contents.find("completed"), std::string::npos);
    EXPECT_NE(contents.find("\"requested\": \"disk\""), std::string::npos);

    // WHICH rung rode is the ladder's decision for this cap, and where the cap
    // falls relative to the in-memory charges is a property of the build's own
    // plan rather than of the route: on the linux-x86 gcc leg a fitting
    // in-memory rung rides at 0.1 GiB - the record says so in its own words,
    // outcome "ladder_fit" with mode "kFastPath" (2026-09-19, CI run
    // 35451596080) - where the fixture was calibrated to refuse (the 2026-09-03
    // probes measured the refusal window against THAT host's plan, which is a
    // reading and not a contract). What the contract fixes is the ORDER, and
    // the record is what discloses it: the disk rung is the ladder's LAST rung,
    // so the knob permits it and never demands it. The assertion therefore
    // follows the record's own outcome, one branch each, rather than the expectation.
    const bool diskRode = contents.find("\"mode\": \"kDisk\"") != std::string::npos;

    if (diskRode)
    {
        // The ladder refused in memory and the storage module's builder rode:
        // 8 n^2 nAux = 8 * 49 * 71 = 27832 is the modeled dense payload of this
        // fixture, a number no other producer of this record writes.
        EXPECT_NE(contents.find("\"disk_bytes\": 27832"), std::string::npos) << contents;

        // The cap child writes the run's serialized energy to the marker
        // (cap_child.cpp writeToken); parse it back for the pin comparison.
        constexpr std::string_view kEnergyKey = "total_energy_hartree";
        const std::string energyPrefix = std::string{"\""} + std::string(kEnergyKey) + "\": ";
        const std::size_t energyPos = contents.find(energyPrefix);
        ASSERT_NE(energyPos, std::string::npos) << "marker: " << contents;
        const std::size_t valueStart = energyPos + energyPrefix.size();
        const std::size_t lineEnd = contents.find('\n', valueStart);
        const double energy = std::stod(contents.substr(
            valueStart, lineEnd == std::string::npos ? std::string::npos : lineEnd - valueStart));
        EXPECT_NEAR(energy, -74.96369193, 1e-5);
    } else
    {
        // A fitting in-memory rung rode, which is the same ladder's first rule:
        // the knob never preempts one. The disk route wrote no payload, and
        // nothing about the disk rung is claimed on this leg - that is this
        // cell's recorded coverage gap rather than a hidden one: the refusal
        // branch above is the disk route's only end-to-end evidence under a cap,
        // and it is reached only where the build's plan refuses at this cap.
        EXPECT_NE(contents.find("\"outcome\": \"ladder_fit\""), std::string::npos)
            << "an in-memory rung rode, so the record must say so: " << contents;
        EXPECT_NE(contents.find("\"disk_bytes\": 0"), std::string::npos) << contents;
    }

    std::filesystem::remove(marker);
#endif
}

// The knob never preempts a fitting memory rung: with the default roomy
// budget the engine's Create-time estimate admits the fast path, so the
// same knobbed text rides kFastPath in RAM and reproduces the in-memory
// RI-J pin. The in-memory mode record carries the
// disk_bytes term as its zero default - only the disk route's synthesized
// record names a payload. The default 16.0 GiB cap is harmless in-process
// (the same arrangement as every budget-path run above).
TEST(DriverRunTest, RiTensorModeDiskNeverPreemptsAFittingMemoryRung) {
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jfit g shells";
    }

    auto result = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "sto-3g"
aux = "def2-universal-jfit"

[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kTight"
ri_tensor_mode = "disk"

[scf]
max_iterations = 100
energy_tolerance = 1e-8
density_tolerance = 1e-6
use_diis = true
)");
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

    // Measured 2026-09-16 and VESTIGIAL: the cell's ten assertions are the route
    // facts (kFastPath, disk_bytes 0, requested disk, resolved in_memory, outcome
    // ladder_fit, forced false, the ladder's-LAST text) plus the energy at 1e-5.
    // Every one of them holds at the operating default; the route facts are
    // gate-INDEPENDENT (the ladder decision is a Create-time estimate), and the
    // energy moves 3.7e-14 (-74.963691925367 against the tight run's
    // -74.96369192536702) - nine orders inside the 1e-5 band.
    const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, -74.96369193, 1e-5);

    EXPECT_NE(result->find("\"mode\": \"kFastPath\""), std::string::npos);
    EXPECT_NE(result->find("\"disk_bytes\": 0"), std::string::npos);
    // The ladder-selected route is never forced, and the record says so:
    // the forced-disk statement is present on every mode record (schema
    // 18) so a forced cell and this one are distinguishable by one key.
    EXPECT_NE(result->find("\"forced_disk\": false"), std::string::npos);
    // The request/ran pairing names this case as what it is: a
    // `disk` request a fitting memory rung rode. ladder_fit is NOT a
    // demotion - the knob permits the disk rung without demanding it, so
    // reporting this normal run as a fallback would be the record lying in
    // the opposite direction.
    EXPECT_NE(result->find("\"requested\": \"disk\""), std::string::npos);
    EXPECT_NE(result->find("\"resolved\": \"in_memory\""), std::string::npos);
    EXPECT_NE(result->find("\"outcome\": \"ladder_fit\""), std::string::npos);
    EXPECT_NE(result->find("\"forced\": false"), std::string::npos);
    EXPECT_NE(result->find("the disk rung is the ladder's LAST"), std::string::npos);
}

// The FORCED disk mode ([diagnostics] force_disk_ri = true, the separate key
// that replaced the retired method.ri_tensor_mode = "forced_disk" word - a
// force is not a rung selection): the diagnostic override that bypasses the
// in-memory ladder entirely, so a benchmark cell can measure the disk-backed
// builder WITHOUT an artificial memory cap. The fixture is the ladder
// fixture's shape at the DEFAULT resource cap: the in-memory ladder FITS
// there - the neighbour test pins kFastPath for the same fixture with "disk" -
// so a run that still ends up on the disk route is proof the key reached it.
// This is the acceptance criterion of the mode: were the key ignored (or read
// as "auto"/"disk"), the run would ride kFastPath in RAM while a cell labelled
// disk recorded it, which is the wrong label the mode exists to prevent. The
// discriminating record facts are the driver-synthesized disk record's own
// (mode kDisk + the dense payload + zeroed engine terms, none of which an
// engine rung decision can produce) and the forced_disk statement that
// separates this cell from a ladder-selected disk run.
//
// The control leg asserts the OTHER direction on the same fixture: with the
// block absent the run rides kFastPath, so the record facts above are
// attributable to the key rather than to the fixture. An energy-only test
// cannot do this job - the two builders agree inside the SCF pin's tolerance,
// so a run that silently swapped builders would still pass an energy check
// while every fact above went false.
TEST(DriverRunTest, RiTensorModeForcedDiskBypassesAFittingLadder) {
#if !defined(QcxHasStorage)
    GTEST_SKIP() << "storage module not built (QCX_ENABLE_IO=OFF): the forced disk rung IS the "
                    "storage module's disk-backed builder, so there is no rung to measure";
#else
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jfit g shells";
    }

    auto result = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "sto-3g"
aux = "def2-universal-jfit"

[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kTight"

[diagnostics]
force_disk_ri = true

[scf]
max_iterations = 100
energy_tolerance = 1e-8
density_tolerance = 1e-6
use_diis = true
)");
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);

    // Measured 2026-09-16 and VESTIGIAL: force_disk_ri bypasses the in-memory
    // ladder entirely, so every discriminating fact this cell exists for (kDisk,
    // disk_bytes 27832, tensor_bytes 0, forced_disk true, honoured, forced true)
    // is gate-INDEPENDENT - it is the key's doing, not the gate's. At the operating
    // default all ten assertions hold and the energy is -74.96369192536702, the
    // same value the tight pair gives, 4.6e-9 from the pin against a 1e-5 band.

    // The disk route's SCF, at the storage suite's RI-J pin
    // (disk_ri_fock_build.cpp).
    const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, -74.96369193, 1e-5);

    // The driver-synthesized disk record: kDisk comes from no engine, so
    // only the disk route (the one code path that constructs
    // DiskRiFockBuilder) can have written it; 8 * 49 * 71 = 27832 is the
    // modeled dense payload of this fixture, and the engine's own terms
    // stay zero on a record no engine decision produced.
    EXPECT_NE(result->find("\"mode\": \"kDisk\""), std::string::npos);
    EXPECT_NE(result->find("\"disk_bytes\": 27832"), std::string::npos);
    EXPECT_NE(result->find("\"tensor_bytes\": 0"), std::string::npos);
    // The statement that makes the cell's label honest: forced, not
    // ladder-selected.
    EXPECT_NE(result->find("\"forced_disk\": true"), std::string::npos);
    // ... and the request/ran pairing (it names the KEY
    // the request came from): the honoured case, so no reason is written
    // (the null honesty policy), and `forced` is true because the request
    // arrived through [diagnostics] force_disk_ri and not through the rung
    // selector.
    EXPECT_NE(result->find("\"requested\": \"forced_disk\""), std::string::npos);
    EXPECT_NE(result->find("\"resolved\": \"disk\""), std::string::npos);
    EXPECT_NE(result->find("\"outcome\": \"honoured\""), std::string::npos);
    EXPECT_NE(result->find("\"forced\": true"), std::string::npos);

    // The control: the identical fixture with no [diagnostics] block. It
    // rides the in-memory ladder's fast path (the ladder fits at the default
    // cap - that is what makes the leg above discriminating), so the disk
    // facts there are the key's doing. The energy is the same within the pin
    // above, which is exactly why the energy alone is not the instrument.
    auto control = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "sto-3g"
aux = "def2-universal-jfit"

[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kTight"

[scf]
max_iterations = 100
energy_tolerance = 1e-8
density_tolerance = 1e-6
use_diis = true
)");
    // Measured 2026-09-16 and VESTIGIAL: this is the forced-disk leg's control on
    // the same fixture, and its facts are gate-independent too. At the operating
    // default it rides kFastPath with disk_bytes 0, forced_disk false, no
    // ri_tensor_mode key, and an energy inside the 1e-5 pin - the same six
    // assertions the 1e-10 pair satisfied.
    ASSERT_TRUE(control.has_value()) << control.error().message;
    EXPECT_NE(control->find("\"converged\": true"), std::string::npos);
    EXPECT_NE(control->find("\"mode\": \"kFastPath\""), std::string::npos);
    EXPECT_NE(control->find("\"disk_bytes\": 0"), std::string::npos);
    EXPECT_NE(control->find("\"forced_disk\": false"), std::string::npos);
    // No request was named - neither key - so the pairing is ABSENT rather
    // than fabricated as an unforced row (an omitted key is the system's
    // judgement, not a request to record).
    EXPECT_EQ(control->find("\"ri_tensor_mode\""), std::string::npos);

    const auto controlTotal = JsonNumber(*control, "\"total_energy_hartree\"");
    ASSERT_TRUE(controlTotal.has_value());
    EXPECT_NEAR(*controlTotal, -74.96369193, 1e-5);
#endif
}

// The [symmetry] full_group = false switch at the driver level:
// the stage is off, so the run JSON carries no symmetry block - the
// output-absence path, pinned here (the scf-level bit-identity of the off
// switch is pinned by RhfLabelingToggleOffIsBitIdenticalAndEnergyInvariant).
// The input is the neighbor test's shape with the switch flipped; where
// the default-true run emits a block, this run emits none.
TEST(DriverRunTest, FullGroupOffOmitsTheSymmetryBlock) {
    // The ri_j_link fixture needs the universal-J aux f shells (L = 3); the
    // lmax=2 CI configs (wsl-gcc/wsl-clang) cannot build them - skip there,
    // like the RiJLinkAutoSelectsTheAuxBasis neighbor. Small run, so no
    // fast-mode guard: MSVC Debug keeps exercising the switch.
    if (!qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "the universal-J aux f shells exceed this build's kMaxEngineL (CI lmax=2)";
    }

    auto result = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "def2-svp"

[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kLoose"

[symmetry]
full_group = false
)");
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NE(result->find("\"converged\": true"), std::string::npos);
    // The stage did not run: no symmetry block is emitted (never a
    // fabricated C1 record), unlike the default-true run next door.
    EXPECT_EQ(result->find("\"symmetry\""), std::string::npos);
}

// The memory seam refuses a run whose modeled peak clears the cap, and
// the diagnostic names the reinstatement ladder — the disk-backed store is
// always named. The H2 direct run's actual usage is ~6 MB (measured), so
// in the CUDA-less builds the 0.045 GiB cap lets the child reach the seam.
// Runs in the cap-child, never in this test process: a job-object cap
// applies to the whole process, and a test that capped its own host binary
// would leave every later test unable to allocate (the full-suite crash
// this rewrite replaced, 2026-08-30).
//
// THE CONTRACT FLIPPED (the 2026-09-17 owner ruling), and this cell's own
// numbers are the reason: the model charged 0.056 GiB of base where the whole
// run commits ~6 MB, and the refusal it raised blocked a run that FITS. A
// prediction that reads too high has no backstop, so under a real cap it is a
// warning now and the run proceeds (the reinstatement ladder still reaches
// stderr, in full, on the advisory line). Measured with the ruling in force:
// this run COMPLETES under the 0.045 GiB cap - exit 0, "completed" - where the
// pre-ruling binaries refused at the seam.
// The CUDA/Debug legs skip as before: their load-time footprint exceeds the
// cap, which is a fact about the image, not about the ruling.
TEST(DriverErrorTest, OverCapDirectRunProceedsUnderTheRealCap) {
#if defined(QcxHasCuda) || defined(_DEBUG)
    // CUDA build only: the cuBLAS runtime's load-time footprint
    // (~0.25-0.3 GiB, measured for the cap probe gate; the environmental floor) exceeds any
    // small cap, so a CUDA-linked child dies inside the runtime before the
    // seam - no qcx ordering can fire first, the driver's probe
    // gate never even runs below the floor. The Debug leg skips the same floor: the Debug exe's
    // load-time footprint also exceeds the tiny cap (the 2026-09-08 wedge class). The
    // advisory contract is CPU-side; the CUDA-less lanes carry this fixture's coverage.
    GTEST_SKIP()
        << "CUDA or Debug build: the runtime's load-time footprint (~0.25-0.3 GiB) exceeds "
           "the 0.045 GiB cap; the advisory contract is verified on the CUDA-less "
           "builds";
#else
    const auto marker = qcx::testing::CapChildMarkerPath(10);
    std::filesystem::remove(marker);

    const int exitCode = qcx::testing::RunCapChild(0.045, marker);

    EXPECT_EQ(exitCode, 0)
        << "the run must proceed on the cap's real limit, not refuse on the model's number";
    const std::string contents = qcx::testing::ReadCapChildMarker(marker);
    EXPECT_NE(contents.find("completed"), std::string::npos)
        << "the H2 direct run fits the 0.045 GiB cap (its own measured usage is ~6 MB): "
        << contents;
    std::filesystem::remove(marker);
#endif
}

// The ri_j budget path (the budget path): the driver grants the engine a
// cap-minus-base workspace budget and withheld it when the cap did not clear
// the modeled base term (capacity zero). H2O/def2-SVP models 0.056 GiB base
// at n = 24, so the 0.045 GiB cap leaves a zero budget.
//
// THE CONTRACT FLIPPED (the 2026-09-17 owner ruling): the withheld budget is
// the same prediction-driven verdict as the direct family's, so under a real
// cap it warns and the run proceeds - on the legacy null-budget path, which
// is what a zero cap always took - with the OS stopping it at the cap instead
// of the model stopping it at zero. Measured with the ruling in force: this
// run COMPLETES under the 0.045 GiB cap (exit 0, "completed"), which is also
// the sharpest evidence for the ruling at this fixture - the model's 0.056
// GiB base exceeds a cap the run fits inside.
// The CUDA/Debug legs skip as before: their load-time footprint exceeds the cap.
TEST(DriverErrorTest, RiJProceedsWhenTheCapDoesNotClearTheBase) {
#if defined(QcxHasCuda) || defined(_DEBUG)
    // CUDA build only: same floor as the direct case above - the
    // runtime's load-time footprint (~0.25-0.3 GiB) exceeds any small
    // cap, so the CUDA-linked child dies before the budget seam .
    // The advisory contract is CPU-side; the CUDA-less lanes carry
    // this fixture's coverage.
    GTEST_SKIP()
        << "CUDA or Debug build: the runtime's load-time footprint (~0.25-0.3 GiB) exceeds "
           "the 0.045 GiB cap; the advisory contract is verified on the "
           "CUDA-less builds";
#else
    // The child's fixture is H2O/def2-svp on the ri_j arm, whose auto-selected
    // universal-J aux carries f shells. Under the 2026-09-17 ruling the over-cap
    // prediction no longer refuses at the seam, so this run DOES reach the
    // engine - and on a build whose 2e engines stop below L = 3 the child
    // refuses with "shell angular momentum exceeds kMaxEngineL of this build"
    // (the pre-ruling refusal at cap_child.cpp shielded these shells, which is
    // what its own comment records). The cap is then not the subject at all:
    // skip like the sibling RiJkGrantSelectsTheBlockedRungWhenTheFastRungDoes-
    // NotFit, whose propane fixture carries the same shells.
    if (!qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "the universal-J aux f shells exceed this build's kMaxEngineL (CI lmax=2)";
    }

    const auto marker = qcx::testing::CapChildMarkerPath(11);
    std::filesystem::remove(marker);

    const int exitCode = qcx::testing::RunCapChild(0.045, marker, "ri_j");

    EXPECT_EQ(exitCode, 0)
        << "the run must proceed on the cap's real limit, not refuse on the model's number";
    const std::string contents = qcx::testing::ReadCapChildMarker(marker);
    EXPECT_NE(contents.find("completed"), std::string::npos)
        << "the ri_j H2O run fits the 0.045 GiB cap its own model refuses it at: " << contents;
    std::filesystem::remove(marker);
#endif
}

// The orbit-expansion request ON THE ri_jk FAMILY (the mechanism key): the
// driver CARRIES the request to the
// composed full-RI builder, which refuses it BY NAME.
//
// The side is decided by what the key NAMES (the key split, owner-ruled
// 2026-09-13): method.ri_orbit_expansion names a MECHANISM - a point-group
// reduction that decides what the run evaluates - so on a family that cannot
// honour it the run refuses rather than dropping the key and running the plain
// walk under the request. A size hint is the other side of that split and is
// dropped with the record stating so (method.ri_chunk_bytes).
//
// The refusal itself is the AM contract, not driver prose: integrals
// RiFullFockBuilder::Create refuses a non-null symmetryReduction /
// auxSymmetryReduction with kUnimplemented, because its 3-center tensor is
// counted but never reduced - accepted-and-silently-ignored is the disclosure rule
// substitution that refusal exists to prevent. What this row changes is that
// the refusal is now REACHABLE from a run: before it, the branch supplied no
// reduction to this family, so the request never arrived anywhere that could
// honour or refuse it, and the record disclosed `not_applicable` over a run
// that computed the plain walk.
//
// FALSIFIABILITY: remove the carrier on the kRiJk arm (the two
// BuildSymmetryReduction calls and the two option assignments) and the first
// cell stops refusing - the run converges on the plain walk with the key
// dropped, which is what the second cell asserts a run WITHOUT the key does;
// the message assertions then fail too, because nothing names the field.
// The control keeps the refusal attributable to the REQUEST rather than to
// ri_jk being broken: the same fixture with no key converges and reports the
// family word. The load-bearing assertion is the FIELD NAME in the refusal,
// not the error code - an unrelated kUnimplemented cannot produce it.
TEST(DriverErrorTest, RiJkCarriesTheOrbitExpansionRequestAndTheBuilderRefusesByName) {
    const auto requested = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "def2-svp"

[method]
type = "rhf"
fock_builder = "ri_jk"
accuracy = "kLoose"
ri_orbit_expansion = true
)");

    ASSERT_FALSE(requested.has_value())
        << "the reduction request was dropped instead of carried to the builder";
    EXPECT_EQ(requested.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(requested.error().message.find("symmetryReduction"), std::string::npos)
        << requested.error().message;
    EXPECT_NE(requested.error().message.find("auxSymmetryReduction"), std::string::npos)
        << requested.error().message;

    // The SAME request on the UNRESTRICTED leg (this wiring's runner), which
    // carries it into its own creation of the same builder: the refusal is the
    // FAMILY's, so both legs must produce it, and a UHF arm that dropped the
    // carrier would run the plain walk under a request that named a mechanism
    // - the defect the restricted cell above exists for, on the leg that did
    // not exist when it was written. The runner's own comment says this is the
    // one place the unrestricted leg inherits the refusal rather than
    // re-deciding it; this cell is what makes that a measurement.
    const auto requestedUhf = RunInputText(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "def2-svp"

[method]
type = "uhf"
fock_builder = "ri_jk"
accuracy = "kLoose"
ri_orbit_expansion = true
)");
    ASSERT_FALSE(requestedUhf.has_value())
        << "the unrestricted leg dropped the reduction request instead of carrying it";
    EXPECT_EQ(requestedUhf.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(requestedUhf.error().message.find("symmetryReduction"), std::string::npos)
        << requestedUhf.error().message;
    EXPECT_NE(requestedUhf.error().message.find("auxSymmetryReduction"), std::string::npos)
        << requestedUhf.error().message;

    // The control: the same run without the key converges on the plain walk,
    // so the refusal above is the request's and not the family's.
    if (qcx::integrals::SupportsL(3))
    {
        const auto control = RunInputText(H2oDef2SvpToml("ri_jk").c_str());
        ASSERT_TRUE(control.has_value()) << control.error().message;
        EXPECT_NE(control->find("\"converged\": true"), std::string::npos);
        EXPECT_NE(control->find("\"builder\": \"ri_jk\""), std::string::npos);
        // The request block is absent when the key was not named - the third
        // reader state of the null-honesty rule, here on the family this row
        // wires the carrier for.
        EXPECT_EQ(control->find("ri_orbit_expansion"), std::string::npos);
    }
}

// The ri_jk budget path (the RI-K memory ladder's driver half, plan section
// 2.4): the driver grants the engine a workspace budget and the engine's
// Create-time estimate decides the rung. Before this the branch handed Create
// no budget, so the estimate that knows this builder's two classes never ran.
//
// THE CONTRACT FLIPPED (the 2026-09-17 owner ruling, the ri_jk half of the
// flip e67e5435 landed for the direct family and ri_j), and this cell's own
// history is why it is the LAST of the three to carry the flip: the model's
// base term for this fixture (0.056 GiB at n = 24) withheld a zero grant and
// the run refused at the seam. The model is deleted (bb8e88b3), and with it
// the refusal: the grant is now the raw cap - WorkspaceBudget::Create(cap) in
// run_driver.cpp's ri_jk arm, no base subtracted - so the cap the model
// refused this family at no longer stops it, and the run PROCEEDS on the
// engine's own rung decision, with the OS enforcing the cap (the ruling's
// contract, stated verbatim on the two sibling rows).
//
// THE CAP IS DERIVED, and every number below is measured with the cap-child
// rebuilt from this tree (a stale child makes these rows lie). The row's old
// 0.045 GiB (48,318,382 B) does NOT work any more, and that
// is not a stale-pin artifact: it now sits AT the run's own peak, where the
// child dies 0xC0000409 with an empty marker - restricted 4/4 runs, UHF 4/4 -
// because the budget the run is granted sizes its arenas, so a bigger grant
// raises the peak with it. Measured band, both legs, this tree:
//   0.044 GiB dies (restricted 3/3, UHF 2/3); 0.045 GiB dies (restricted 4/4,
//   UHF 4/4); 0.046 and 0.047 GiB complete on both legs; 0.05 GiB completes
//   restricted 6/6 and UHF 4/4.
// 0.05 GiB (53,687,091 B) is therefore the smallest cap probed with stable
// completions, ~5.4 MB clear of the death edge. It is ALSO still below the
// deleted model's 0.056 GiB base (60,129,542 B), so the row keeps sitting on
// the cap the family was refused at - the ruling's case, in miniature.
//
// The pre-gate setup admission is checked against the same cap and does NOT
// decide this row: it refuses when 1.5 x SetupRampPeakBytes exceeds the cap,
// and it printed no text at the 0.044 GiB (47,244,902 B) probe - so its
// threshold for this fixture is at or below 47,244,902 B and the 0.05 GiB cap
// clears it with room. (The arm's own number on a fixture big enough to print
// it: propane/def2-SVP, 158,987,464 B, threshold 238,481,196 B.)
//
// FALSIFIABILITY: drop the grant carrier (pass no budget) and the run takes the
// legacy null-budget path; the marker still records "completed", so the exit
// code alone is not the pin - the rung and tensor cells below are, and they are
// the engine's own record rather than driver arithmetic.
TEST(DriverErrorTest, RiJkProceedsWhenTheCapDoesNotClearTheBase) {
#if defined(QcxHasCuda) || defined(_DEBUG)
    // Same floor as the direct and ri_j cases above: the CUDA runtime's
    // load-time footprint (~0.25-0.3 GiB) exceeds any small cap, and the
    // Debug exe's does too (the 2026-09-08 wedge class). The advisory
    // contract is CPU-side; the CUDA-less lanes carry this fixture.
    GTEST_SKIP()
        << "CUDA or Debug build: the runtime's load-time footprint (~0.25-0.3 GiB) exceeds "
           "the 0.05 GiB cap; the advisory contract is verified on the "
           "CUDA-less builds";
#else
    // The child's fixture is H2O/def2-svp on the ri_jk arm, whose auto-selected
    // JK fit carries f shells - the same shells the sibling
    // RiJkGrantSelectsTheBlockedRungWhenTheFastRungDoesNotFit skips on, and the
    // reason that cell carries no budget subject here either: on a build whose
    // 2e engines stop below L = 3 the child refuses with "shell angular momentum
    // exceeds kMaxEngineL of this build" before the cap can decide anything.
    if (!qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "the JK-fit aux f shells exceed this build's kMaxEngineL (CI lmax=2)";
    }

    const auto marker = qcx::testing::CapChildMarkerPath(17);
    std::filesystem::remove(marker);

    // The derived cap (the comment above carries the band): 0.05 GiB, the
    // smallest probed cap with stable completions on BOTH legs, still under
    // the deleted model's 0.056 GiB base for this fixture.
    constexpr double kProceedingCapGiB = 0.05;
    const int exitCode = qcx::testing::RunCapChild(kProceedingCapGiB, marker, "ri_jk");

    EXPECT_EQ(exitCode, 0)
        << "the run must proceed on the cap's real limit, not refuse on the model's number";
    const std::string contents = qcx::testing::ReadCapChildMarker(marker);
    EXPECT_NE(contents.find("completed"), std::string::npos)
        << "the ri_jk H2O run fits the 0.05 GiB cap its deleted model refuses it at: " << contents;
    // The engine's own rung decision, carried out rather than recomputed: the
    // grant is the cap, the fast rung's arena is the grant less its fixed
    // charge, so the fast rung rides - which is what the model's withheld
    // grant used to make impossible.
    EXPECT_NE(contents.find("\"rung\": \"kFast\""), std::string::npos) << contents;
    // The tensor term ties the block to the fixture's arithmetic: 8 n^2 nAux
    // at n = 24, nAux = 113 - a number no other producer of this record writes.
    EXPECT_NE(contents.find("\"tensor_bytes\": 520704"), std::string::npos) << contents;
    std::filesystem::remove(marker);

    // The same decision on the UNRESTRICTED leg, through the child's mirror
    // fixture (ri_jk_uhf): the runner owns a COPY of the budget path - its own
    // grant arithmetic, its own rung decision - so the restricted cell above
    // cannot stand for it. The same pair of readings on both legs (the rung and
    // the tensor term) is what makes the two cells a comparison rather than a
    // copy.
    const auto uhfMarker = qcx::testing::CapChildMarkerPath(24);
    std::filesystem::remove(uhfMarker);

    const int uhfExitCode = qcx::testing::RunCapChild(kProceedingCapGiB, uhfMarker, "ri_jk_uhf");

    EXPECT_EQ(uhfExitCode, 0) << "the unrestricted run must proceed on the cap's real limit";
    const std::string uhfContents = qcx::testing::ReadCapChildMarker(uhfMarker);
    EXPECT_NE(uhfContents.find("completed"), std::string::npos) << uhfContents;
    EXPECT_NE(uhfContents.find("\"rung\": \"kFast\""), std::string::npos) << uhfContents;
    EXPECT_NE(uhfContents.find("\"tensor_bytes\": 520704"), std::string::npos) << uhfContents;
    std::filesystem::remove(uhfMarker);
#endif
}

// The rung half of the same budget path: a cap that leaves a grant
// too small for the FAST rung's class and large enough for the BLOCKED one
// makes a RUN select RiFullFockRung::kBlocked. That rung was dead code from a
// run's point of view before this wiring - the branch admitted or refused the
// family whole against the RI-J envelope (3 x tensor + base) and never handed
// Create a budget to decide with - and the observation is the engine's own
// record, not driver arithmetic: the marker carries the ri_jk_mode block's
// rung word and its tensor term, so the row pins WHICH rung rode rather than
// that the run completed.
//
// The fixture is propane/def2-SVP with the auto-selected JK fit (n = 82,
// nAux = 369, tensor = 8 n^2 nAux = 19,849,248 B). Propane rather than water
// because the band has to clear this child's real footprint to be observable
// at all: at n = 24 the model's base term (0.056 GiB) is BELOW the run's real
// commit, so every cap in the band throttles the job object - MEASURED at HEAD
// on the ri_jk fixture: 0.0575-0.0600 GiB never completed in 150 s, while the
// same run at a cap above the band finished in 2 s on the fast rung. At n = 82
// the modeled base (0.2229 GiB) carries the direct family's screened-quartet
// working set, which this path never allocates, and the grant inside the band
// leaves room for the blocked rung's slice.
//
// The band is MEASURED on this fixture (2026-09-16, this child, one run per
// cap): 0.245 GiB refuses at Create (the grant does not fit the blocked
// rung's class either), 0.255-0.270 GiB select kBlocked, 0.272 GiB and above
// select kFast. The cell pins 0.255 because it sits between the edges rather
// than on one, and the control pins 0.35 for the same reason. A future band
// move fails this row at the rung assertion, with the measured band above as
// the thing to re-measure - and the blocked rung's wall is carried too: it is
// ~20x the fast rung's on this fixture (measured 52-67 s against 3 s), which
// is the one cost of this row.
//
// FALSIFIABILITY: remove the budget carrier (pass nullptr, keep the grant
// arithmetic) and the 0.255 GiB cell runs on the FAST rung - the block reports
// "kFast" and the rung assertion fails; remove the grant entirely and the same
// cell refuses instead of completing. The control is the same fixture at a cap
// whose grant fits both classes (0.35 GiB): the fast rung rides there, so the
// band cap is what selected the blocked one, not the fixture.
TEST(DriverErrorTest, RiJkGrantSelectsTheBlockedRungWhenTheFastRungDoesNotFit) {
    if (!qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "the JK-fit aux f shells exceed this build's kMaxEngineL (CI lmax=2)";
    }

    // The band cap: the grant is cap minus the modeled base (0.2229 GiB), i.e.
    // 34.5 MB - above the blocked rung's class (its fixed part plus one
    // 19,849,248 B tensor copy plus the shell-range slice) and below the fast
    // rung's (the same fixed part plus TWO tensor copies plus the arena).
    const auto bandMarker = qcx::testing::CapChildMarkerPath(18);
    std::filesystem::remove(bandMarker);

    const int bandExit = qcx::testing::RunCapChild(0.255, bandMarker, "ri_jk_propane");

    EXPECT_EQ(bandExit, 0) << "the run must complete on the blocked rung, not refuse";
    const std::string bandContents = qcx::testing::ReadCapChildMarker(bandMarker);
    EXPECT_NE(bandContents.find("\"rung\": \"kBlocked\""), std::string::npos) << bandContents;
    // The tensor term ties the block to the fixture's arithmetic: 8 n^2 nAux at
    // n = 82, nAux = 369 - a number no other producer of this record writes.
    EXPECT_NE(bandContents.find("\"tensor_bytes\": 19849248"), std::string::npos) << bandContents;
    std::filesystem::remove(bandMarker);

    // The control: a cap whose grant fits the fast rung's class too. It keeps
    // the band cell attributable to the GRANT rather than to the fixture.
    const auto fastMarker = qcx::testing::CapChildMarkerPath(19);
    std::filesystem::remove(fastMarker);

    const int fastExit = qcx::testing::RunCapChild(0.35, fastMarker, "ri_jk_propane");

    EXPECT_EQ(fastExit, 0) << "the control cap must clear the fast rung";
    const std::string fastContents = qcx::testing::ReadCapChildMarker(fastMarker);
    EXPECT_NE(fastContents.find("\"rung\": \"kFast\""), std::string::npos) << fastContents;
    std::filesystem::remove(fastMarker);
}

// The pair-class admission gate (the pair-class root fix, 2026-08-31,
// The adaptive seam; the materialization accounting, 2026-09-13): the class path's
// Create-time charge is the MINIMUM of two never-under upper bounds of the
// same allocation - the eager singleton-orbit ceiling (ClassTableBytes, the
// full canonical member-quartet space) and the bound of the shape the path
// actually materializes (ClassMaterializationBytes, whose counts come from
// CountClassMaterialization without materializing anything). The 24-water
// cluster fixture (120 shells, 7,260 canonical pairs) carries the C2v/D2h
// structure, so the class path requests engagement. The 5.2 GiB cap's
// cap-minus-base budget cannot fit the ~6.3 GiB ceiling, and under the
// ceiling ALONE the gate DISENGAGED here - the state this test used to pin.
// With the accounting fixed the charge is the materialization bound, which
// the budget clears by orders of magnitude, so the class path ENGAGES and the
// run completes through it (or refuses with the ladder when the rest of the
// stack cannot fit) - the exit code is 0 or 1, never the -1073740791
// fastfail. Runs in the cap-child like the other cap tests (a job-object cap
// applies to the whole process).
TEST(DriverErrorTest, ClassTableAdmissionGateEngagesOnTheMaterializedShape) {
    const auto marker = qcx::testing::CapChildMarkerPath(12);
    std::filesystem::remove(marker);

    const int exitCode = qcx::testing::RunCapChild(5.2, marker, "water_cluster");

    // The pre-fix 0xC0000409 fastfail (int -1073740791) must never return.
    EXPECT_NE(exitCode, -1073740791) << "the pre-fix heap-growth death must never return";
    EXPECT_TRUE(exitCode == 0 || exitCode == 1)
        << "the run must complete through the class path or refuse with the ladder";

    const std::string contents = qcx::testing::ReadCapChildMarker(marker);

    if (exitCode == 0)
    {
        // The completed leg: the class path engaged on the shape it actually
        // materializes - the disengagement flag clear, and a charge that is a
        // materialization bound rather than the ~6.3 GiB ceiling this query
        // used to be refused by.
        EXPECT_NE(contents.find("completed"), std::string::npos);
        EXPECT_NE(contents.find("\"class_path_disengaged\": false"), std::string::npos);
        const auto charge = JsonNumber(contents, "\"class_table_bytes\"");
        ASSERT_TRUE(charge.has_value()) << "the record carries no class_table_bytes";
        // WHAT IS ASSERTED IS THE DECISION: the gate engaged on the shape this
        // path actually materializes, so the charge is a materialization
        // bound and not the ~6.3 GiB eager ceiling that used to refuse the
        // query. A magnitude is part of the contract only relative to the two
        // quantities the decision turns on - the cap it must fit inside and
        // the ceiling it must be below - so it is bounded against those, never
        // pinned as a byte count. The count is measured on the same child: a
        // direct run of this cap-child on this input gave 23,723,568 B = 22.6
        // MiB in Release and 24,919,824 B in the windows-msvc Debug leg
        // (2026-09-19, CI run 35451596080) - 5.0 % apart on the SAME layout,
        // which is the build configuration (_ITERATOR_DEBUG_LEVEL and checked
        // iterators inflating the containers) and not the gate. The fixture
        // itself fixes the scale: the detector's reduction and the Schwarz
        // screening decide the counts, so any absolute pin here is a pin on
        // one build's allocator.
        EXPECT_LT(*charge, 1.0e9) << "the charge is a materialization bound, not the eager ceiling";
        EXPECT_LT(*charge, 5.2 * 1024.0 * 1024.0 * 1024.0 * 0.01)
            << "the charge must sit two orders inside the cap it was admitted under";
    } else
    {
        EXPECT_NE(contents.find("refused: "), std::string::npos);
        EXPECT_NE(contents.find("direct screened"), std::string::npos);
    }

    std::filesystem::remove(marker);
}

// The full [properties] block through the driver path -
// the charges/ESP/EDDB/Nalewajski pins of the properties module tests
// (charges_test.cpp, esp_test.cpp, eddb_test.cpp, nalewajski_test.cpp),
// now emitted in the run JSON. The grid is the driver's pinned 80 x 194
// and the promolecular SAD fragments are built identically to the charges
// tests' fixture, so the pins carry the module tolerances; the density-
// sensitive analyses inherit the direct builder's kTight density (the
// H2O energy pin above is 1e-5), which the loose pins absorb.
TEST(DriverPinTest, H2oDirectRhfStage72BlocksPins) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kH2oStage72Toml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Hirshfeld: [H, H, O] atom order; the pyscf cross-check pins.
    const auto hirshfeld = JsonNumberArray(*result, "\"hirshfeld\":");
    ASSERT_TRUE(hirshfeld.has_value());
    ASSERT_EQ(hirshfeld->values.size(), 3u);
    EXPECT_NEAR(hirshfeld->values[0], 0.22279, 2e-2);
    EXPECT_NEAR(hirshfeld->values[1], 0.22279, 2e-2);
    EXPECT_NEAR(hirshfeld->values[2], -0.44558, 2e-2);

    // Voronoi: the O cell charges positively (the pyscf pattern).
    const auto voronoi = JsonNumberArray(*result, "\"voronoi\":");
    ASSERT_TRUE(voronoi.has_value());
    ASSERT_EQ(voronoi->values.size(), 3u);
    EXPECT_NEAR(voronoi->values[0], -0.32928, 2e-2);
    EXPECT_NEAR(voronoi->values[1], -0.32928, 2e-2);
    EXPECT_NEAR(voronoi->values[2], 0.65282, 2e-2);

    // ESP CHELPG: anchored on the "esp" block - the un-anchored
    // "\"charges\":" would hit the hirshfeld/voronoi block first.
    const auto espPos = result->find("\"esp\"");
    ASSERT_NE(espPos, std::string::npos);
    const auto espCharges = JsonNumberArray(*result, "\"charges\":", espPos);
    ASSERT_TRUE(espCharges.has_value());
    ASSERT_EQ(espCharges->values.size(), 3u);
    EXPECT_NEAR(espCharges->values[0], 0.307317743072, 1e-5);
    EXPECT_NEAR(espCharges->values[1], 0.307210416720, 1e-5);
    EXPECT_NEAR(espCharges->values[2], -0.614528159792, 1e-5);

    // EDDB: the runEDDB reference pins.
    const auto eddbTotal = JsonNumber(*result, "\"total_population\"");
    ASSERT_TRUE(eddbTotal.has_value());
    EXPECT_NEAR(*eddbTotal, 0.013065, 1e-4);
    const auto atomicPopulations = JsonNumberArray(*result, "\"atomic_populations\":");
    ASSERT_TRUE(atomicPopulations.has_value());
    ASSERT_EQ(atomicPopulations->values.size(), 3u);
    EXPECT_NEAR(atomicPopulations->values[0], 0.00285, 1e-5);
    EXPECT_NEAR(atomicPopulations->values[1], 0.00285, 1e-5);
    EXPECT_NEAR(atomicPopulations->values[2], 0.00736, 1e-5);

    // Nalewajski-Mrozek: row 0 of the bond-order matrix carries both the
    // O-H and the H-H bond order (atom order [H, H, O]); the valences
    // follow in the same order.
    const auto nalewajskiPos = result->find("\"nalewajski\"");
    ASSERT_NE(nalewajskiPos, std::string::npos);
    const auto bondRow0 = JsonMatrixRow0(*result, "\"bond_orders\":", nalewajskiPos);
    ASSERT_TRUE(bondRow0.has_value());
    ASSERT_EQ(bondRow0->size(), 3u);
    EXPECT_NEAR((*bondRow0)[0], 0.0, 1e-12);
    EXPECT_NEAR((*bondRow0)[1], 6.75636403e-4, 1e-5);
    EXPECT_NEAR((*bondRow0)[2], 1.19996670, 1e-3);
    const auto totalValence = JsonNumberArray(*result, "\"total_valence\":", nalewajskiPos);
    ASSERT_TRUE(totalValence.has_value());
    ASSERT_EQ(totalValence->values.size(), 3u);
    EXPECT_NEAR(totalValence->values[0], 0.49197462, 1e-5);
    EXPECT_NEAR(totalValence->values[1], 0.49197462, 1e-5);
    EXPECT_NEAR(totalValence->values[2], 0.98328444, 1e-5);

    // The module pins v_ab's sum (ionic + covalent, the one-center valence,
    // nalewajski_test.cpp) rather than the parts; the driver mirrors both
    // the parts and the sum.
    const auto ionicValence = JsonNumberArray(*result, "\"atomic_ionic_valence\":", nalewajskiPos);
    const auto covalentValence =
        JsonNumberArray(*result, "\"atomic_covalent_valence\":", nalewajskiPos);
    ASSERT_TRUE(ionicValence.has_value());
    ASSERT_TRUE(covalentValence.has_value());
    ASSERT_EQ(ionicValence->values.size(), 3u);
    ASSERT_EQ(covalentValence->values.size(), 3u);
    EXPECT_NEAR(ionicValence->values[0], 0.25401269, 1e-5);
    EXPECT_NEAR(ionicValence->values[1], 0.25401269, 1e-5);
    EXPECT_NEAR(ionicValence->values[2], 0.82306831, 1e-4);
    EXPECT_NEAR(covalentValence->values[0], 0.0, 1e-12);
    EXPECT_NEAR(covalentValence->values[1], 0.0, 1e-12);
    EXPECT_NEAR(covalentValence->values[2], 0.08589850, 1e-4);
    EXPECT_NEAR(ionicValence->values[2] + covalentValence->values[2], 0.90896682, 1e-4);
}

// The H2O ETS-NOCV decomposition through the driver path. The
// fragments are written in FILE order (O alone + the two H atoms) and the
// driver maps them onto the canonical [H, H, O] order: the first fragment
// energy is the isolated O atom's (-73.8 hartree, eight electrons), the
// second the H2 fragment's - the module test's partition {O}, {H2} pins
// (nocv_test.cpp), which are themselves O-first.
TEST(DriverPinTest, H2oDirectRhfNocvPartitionPins) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kH2oNocvToml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Fragment-derived values are exact (the fragment SCFs run dense
    // inside the analysis); the molecular-density-dependent ones carry the
    // direct builder's kTight density error, so their pins relax (the
    // H2O energy pin above is 1e-5).
    const auto fragmentEnergies = JsonNumberArray(*result, "\"fragment_energies\":");
    ASSERT_TRUE(fragmentEnergies.has_value());
    ASSERT_EQ(fragmentEnergies->values.size(), 2u);
    EXPECT_NEAR(fragmentEnergies->values[0], -73.834215719824, 1e-7); // O
    EXPECT_NEAR(fragmentEnergies->values[1], -0.921901167759, 1e-7); // H2

    const auto electrostatic = JsonNumber(*result, "\"electrostatic\":");
    const auto pauli = JsonNumber(*result, "\"pauli\":");
    const auto orbital = JsonNumber(*result, "\"orbital\":");
    ASSERT_TRUE(electrostatic.has_value());
    ASSERT_TRUE(pauli.has_value());
    ASSERT_TRUE(orbital.has_value());
    EXPECT_NEAR(*electrostatic, -0.724475877668, 1e-7);
    EXPECT_NEAR(*pauli, 1.885101541776, 1e-7);
    EXPECT_NEAR(*orbital, -1.367437022962, 1e-4);

    // The partition identity: E_elstat + E_Pauli + E_orb = E_mol -
    // sum_i E_frag_i, exact for the analyzer's own numbers.
    const auto binding = JsonNumber(*result, "\"binding_energy\":");
    ASSERT_TRUE(binding.has_value());
    EXPECT_NEAR(*binding, *electrostatic + *pauli + *orbital, 1e-9);

    // E_orb resolves into the sign-paired channels by construction.
    const auto components = JsonNumberArray(*result, "\"orbital_components\":");
    ASSERT_TRUE(components.has_value());
    ASSERT_EQ(components->values.size(), 4u);
    double componentSum = 0.0;

    for (const double component : components->values)
    {
        componentSum += component;
    }

    EXPECT_NEAR(componentSum, *orbital, 1e-9);
}

// The H2 exactly-paired NOCV channel through the driver path -
// the textbook sigma pairing of the module test: one channel, nu = 1,
// E_orb = the single component, and the spin-resolved channels of the
// unrestricted resolution.
TEST(DriverPinTest, H2DirectRhfNocvExactlyPairedChannel) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto result = RunInputText(kH2NocvToml);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto fragmentEnergies = JsonNumberArray(*result, "\"fragment_energies\":");
    ASSERT_TRUE(fragmentEnergies.has_value());
    ASSERT_EQ(fragmentEnergies->values.size(), 2u);
    EXPECT_NEAR(fragmentEnergies->values[0], -0.470988242977, 1e-9);
    EXPECT_NEAR(fragmentEnergies->values[1], fragmentEnergies->values[0], 1e-12);

    const auto electrostatic = JsonNumber(*result, "\"electrostatic\":");
    const auto pauli = JsonNumber(*result, "\"pauli\":");
    const auto orbital = JsonNumber(*result, "\"orbital\":");
    ASSERT_TRUE(electrostatic.has_value());
    ASSERT_TRUE(pauli.has_value());
    ASSERT_TRUE(orbital.has_value());
    EXPECT_NEAR(*electrostatic, -0.060440900418, 1e-7);
    EXPECT_NEAR(*pauli, 0.904261131022, 1e-7);
    EXPECT_NEAR(*orbital, -1.018558069713, 1e-6);

    // Exactly-paired: one channel carrying the whole E_orb, nu = 1.
    const auto eigenvalues = JsonNumberArray(*result, "\"nocv_eigenvalues\":");
    ASSERT_TRUE(eigenvalues.has_value());
    ASSERT_EQ(eigenvalues->values.size(), 1u);
    EXPECT_NEAR(eigenvalues->values[0], 1.0, 1e-6);
    const auto components = JsonNumberArray(*result, "\"orbital_components\":");
    ASSERT_TRUE(components.has_value());
    ASSERT_EQ(components->values.size(), 1u);
    EXPECT_NEAR(components->values[0], *orbital, 1e-9);

    // The spin-resolved resolution: the physical channels of the open-
    // shell fragment problem (the beta channel carries the bonding).
    const auto orbitalUnrestricted = JsonNumber(*result, "\"orbital_unrestricted\":");
    ASSERT_TRUE(orbitalUnrestricted.has_value());
    EXPECT_NEAR(*orbitalUnrestricted, -0.584906754566, 1e-6);
    const auto alphaComponents = JsonNumberArray(*result, "\"orbital_components_alpha\":");
    ASSERT_TRUE(alphaComponents.has_value());
    ASSERT_EQ(alphaComponents->values.size(), 1u);
    EXPECT_NEAR(alphaComponents->values[0], -0.338485772663, 1e-6);
    const auto alphaNus = JsonNumberArray(*result, "\"nocv_eigenvalues_alpha\":");
    ASSERT_TRUE(alphaNus.has_value());
    ASSERT_EQ(alphaNus->values.size(), 1u);
    EXPECT_NEAR(alphaNus->values[0], 0.0, 1e-6);
    const auto betaComponents = JsonNumberArray(*result, "\"orbital_components_beta\":");
    ASSERT_TRUE(betaComponents.has_value());
    ASSERT_EQ(betaComponents->values.size(), 1u);
    EXPECT_NEAR(betaComponents->values[0], -0.246420981902, 1e-6);
    const auto betaNus = JsonNumberArray(*result, "\"nocv_eigenvalues_beta\":");
    ASSERT_TRUE(betaNus.has_value());
    ASSERT_EQ(betaNus->values.size(), 1u);
    EXPECT_NEAR(betaNus->values[0], 1.0, 1e-6);
}

// The H2O condensed Fukui indices through the driver path - the
// driver runs the UHF doublet anion and cation itself (fukui_test.cpp's
// recipe: RHF neutral, UHF charged species). The defining identities and
// the textbook H2O response: O is the electrophilic site (the lone
// pairs), the H's the nucleophilic one (the sigma* OH orbital).
TEST(DriverPinTest, H2oDirectRhfFukuiSumRules) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "three real SCFs: Release-only";
    }

    // The charged species' UHF runs hover at the default 100-iteration
    // budget under kTight (the species-SCF convergence gate surfaces it; the
    // species SCF is formally not converged at budget end though converged
    // for every observable), so the budget is raised - the same discipline as
    // the NOCV cc-pVDZ fragment test.
    const auto toml = std::string(kH2oFukuiToml) + "[scf]\nmax_iterations = 200\n";
    auto result = RunInputText(toml.c_str());
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto fukuiPos = result->find("\"fukui\"");
    ASSERT_NE(fukuiPos, std::string::npos);
    const auto nucleophilic = JsonNumberArray(*result, "\"nucleophilic\":", fukuiPos);
    const auto electrophilic = JsonNumberArray(*result, "\"electrophilic\":", fukuiPos);
    const auto radical = JsonNumberArray(*result, "\"radical\":", fukuiPos);
    ASSERT_TRUE(nucleophilic.has_value());
    ASSERT_TRUE(electrophilic.has_value());
    ASSERT_TRUE(radical.has_value());
    ASSERT_EQ(nucleophilic->values.size(), 3u);
    ASSERT_EQ(electrophilic->values.size(), 3u);
    ASSERT_EQ(radical->values.size(), 3u);

    double nucleophilicSum = 0.0;
    double electrophilicSum = 0.0;
    double radicalSum = 0.0;

    for (std::size_t i = 0; i < 3; ++i)
    {
        nucleophilicSum += nucleophilic->values[i];
        electrophilicSum += electrophilic->values[i];
        radicalSum += radical->values[i];
    }

    EXPECT_NEAR(nucleophilicSum, 1.0, 1e-8);
    EXPECT_NEAR(electrophilicSum, 1.0, 1e-8);
    EXPECT_NEAR(radicalSum, 1.0, 1e-8);
    EXPECT_GT(nucleophilic->values[0], nucleophilic->values[2]);
    EXPECT_GT(electrophilic->values[2], electrophilic->values[0]);
}

TEST(DriverPinTest, O2UhfFukuiExercisesTheTripletBranches) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "three real SCFs: Release-only";
    }

    // O2 is a triplet (M = 3), so the Fukui cation runs at M - 1 = 2 (the
    // doublet branch) and the anion at M + 1 = 4 - the singlet H2O test
    // above only reaches the cation's M = 1 "else 2" fallback. The
    // condensed Fukui indices still sum to one per species. The species
    // budget is raised past the default-100 hover, as in the H2O test
    // (spliced into the fixture's existing [scf] block - a duplicate table
    // is a TOML parse error).
    std::string toml = kO2Toml;
    const std::size_t useDiis = toml.find("use_diis = true");
    ASSERT_NE(useDiis, std::string::npos);
    toml.insert(useDiis, "max_iterations = 200\n");
    toml += "[properties]\nfukui = true\n";
    auto result = RunInputText(toml.c_str());
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto fukuiPos = result->find("\"fukui\"");
    ASSERT_NE(fukuiPos, std::string::npos);
    const auto nucleophilic = JsonNumberArray(*result, "\"nucleophilic\":", fukuiPos);
    const auto electrophilic = JsonNumberArray(*result, "\"electrophilic\":", fukuiPos);
    const auto radical = JsonNumberArray(*result, "\"radical\":", fukuiPos);
    ASSERT_TRUE(nucleophilic.has_value());
    ASSERT_TRUE(electrophilic.has_value());
    ASSERT_TRUE(radical.has_value());
    ASSERT_EQ(nucleophilic->values.size(), 2u);
    ASSERT_EQ(electrophilic->values.size(), 2u);
    ASSERT_EQ(radical->values.size(), 2u);

    double nucleophilicSum = 0.0;
    double electrophilicSum = 0.0;
    double radicalSum = 0.0;

    for (std::size_t i = 0; i < 2; ++i)
    {
        nucleophilicSum += nucleophilic->values[i];
        electrophilicSum += electrophilic->values[i];
        radicalSum += radical->values[i];
    }

    // Looser than the singlet pin: the near-degenerate UHF fixed-point
    // cluster (the 5e-7 energy pin above) perturbs the condensed indices
    // in their last digits, but the per-species conservation is exact.
    EXPECT_NEAR(nucleophilicSum, 1.0, 1e-6);
    EXPECT_NEAR(electrophilicSum, 1.0, 1e-6);
    EXPECT_NEAR(radicalSum, 1.0, 1e-6);
}

TEST(DriverErrorTest, FukuiRejectsUnconvergedSpeciesScf) {
    // The unconverged-budget refusal (no tiny-maxIterations test): the
    // driver runs the charged Fukui species' UHF SCFs itself,
    // and a species run that exhausts its budget returns the LAST iterate
    // with converged == false. Before the fix, those last-iterate
    // densities fed AnalyzeFukui silently; the run must fail instead of
    // emitting Fukui indices computed from unconverged species states.
    // (The main run's own non-convergence is recorded in the JSON, not an
    // error - the charged species are the harder-converging ones.)
    const auto toml = std::string(kH2oFukuiToml) + "[scf]\nmax_iterations = 1\n";
    auto result = RunInputText(toml.c_str());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("did not converge"), std::string::npos);
}

TEST(DriverErrorTest, RejectsUhfNocvCombination) {
    // The ETS-NOCV analysis is closed-shell in v1; the combination is
    // rejected before the SCF runs.
    const auto uhfNocv = RunInputText(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["O", 0.0, 0.0, 1.2]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"
[properties]
nocv_fragments = [[0], [1]]
)");
    ASSERT_FALSE(uhfNocv.has_value());
    EXPECT_EQ(uhfNocv.error().code, qcx::ErrorCode::kUnimplemented);
}

TEST(DriverErrorTest, CertifiedBoundEnforcementRejectedThroughThePipeline) {
    // `enforce_certified_bound` is a claim - it asserts the fp32 lane was
    // certified against a budget - and only the RHF direct family's
    // MACHINERY member can fold it (fock_build.cpp: the FastPath compares,
    // the LightPath and an engaged ladder refuse by name). Every other route
    // used to accept the key and DROP it, so the request is refused BY NAME
    // at the resolution point instead. The judgement reads the resolved kind
    // and the same within-family predicate the wiring reads, so it cannot
    // disagree with the builder that actually runs.
    //
    // The sharp end is the LEAN member: it is the DEFAULT at nBasis <= 1000,
    // and LeanFockBuildOptions has no enforcement field at all, so the field
    // is dropped by construction rather than by oversight. Both spellings of
    // lean are covered - the absent key that resolves to it at this size, and
    // an explicit fock_builder = "lean".
    const auto leanByDefault = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
enforce_certified_bound = true
)");
    ASSERT_FALSE(leanByDefault.has_value());
    EXPECT_EQ(leanByDefault.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(leanByDefault.error().message.find("method.enforce_certified_bound"),
              std::string::npos);
    EXPECT_NE(leanByDefault.error().message.find("LEAN member"), std::string::npos);

    const auto explicitLean = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "lean"
accuracy = "kNormal"
enforce_certified_bound = true
)");
    ASSERT_FALSE(explicitLean.has_value());
    EXPECT_EQ(explicitLean.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(explicitLean.error().message.find("LEAN member"), std::string::npos);

    // The other families: RiEngineOptions carries no such field either, and
    // the option appears in no RI, QFMM or CUDA translation unit.
    const auto riJLink = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
aux = "def2-universal-jfit"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
enforce_certified_bound = true
)");
    ASSERT_FALSE(riJLink.has_value());
    EXPECT_EQ(riJLink.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(riJLink.error().message.find("ri_j_link"), std::string::npos);

    const auto qfmm = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "qfmm"
accuracy = "kNormal"
enforce_certified_bound = true
)");
    ASSERT_FALSE(qfmm.has_value());
    EXPECT_EQ(qfmm.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(qfmm.error().message.find("qfmm"), std::string::npos);

    // Both UHF legs drop it: the direct-UHF wiring never assigns
    // FockBuildOptions::enforceCertifiedBoundBudget, so no UHF route can
    // report the bound the claim asserts - a UHF run is refused whatever
    // builder it names.
    const auto uhf = RunInputText(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["O", 0.0, 0.0, 1.2]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"
enforce_certified_bound = true
)");
    ASSERT_FALSE(uhf.has_value());
    EXPECT_EQ(uhf.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(uhf.error().message.find("UHF"), std::string::npos);

    // The Kohn-Sham word on the same leg, and the reason the refusal has to
    // name the METHOD rather than its shape: "an unrestricted run" is true of
    // two different words now that UKS exists, so a message that stopped at
    // the shape would leave an author unable to tell which of them they hit.
    // The enforcement itself travels to the direct family's options struct on
    // the RHF wiring alone, so UKS drops the request exactly as UHF does.
    const auto uks = RunInputText(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["O", 0.0, 0.0, 1.2]]
[basis]
orbital = "sto-3g"
[method]
type = "uks"
functional = "slater"
fock_builder = "direct"
accuracy = "kNormal"
enforce_certified_bound = true
)");
    ASSERT_FALSE(uks.has_value());
    EXPECT_EQ(uks.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(uks.error().message.find("UKS"), std::string::npos);

    // And the route that CAN honour it passes this check - the refusal is
    // not indiscriminate: an explicit fock_builder = "direct" reaches the
    // machinery member at ANY size (the lean ceiling constrains only the
    // absent key), so the same molecule refused above is admitted here. What
    // is pinned is that the ENFORCEMENT did not refuse it: the run may still
    // fail for its own reasons (this member's own rungs refuse an engaged
    // ladder or a LightPath from inside BuildFock), and those failures are a
    // different, already-tested contract.
    const auto machinery = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
enforce_certified_bound = true
)");

    if (!machinery.has_value())
    {
        EXPECT_EQ(machinery.error().message.find("method.enforce_certified_bound"),
                  std::string::npos);
    }
}

TEST(DriverErrorTest, ForcedCertifiedLaneRejectedOnTheRoutesWithoutOne) {
    // `force_certified_lane` is the same class of claim as the enforcement key
    // above - it asserts the fp32 lane ran - and the request travels in
    // exactly one field, FockBuildOptions::useCertifiedMixedPrecision, which
    // only the direct family's MACHINERY members assemble. Every other route
    // used to be a place the key could have been accepted and dropped, so each
    // is refused BY NAME at the resolution point instead, judged on the
    // RESOLVED kind and the same within-family predicate the wiring reads.
    //
    // The sharp end is again the LEAN member: it is the DEFAULT at nBasis <=
    // 1000 and LeanFockBuildOptions has no fp32 lane and no such field at all.
    const auto leanByDefault = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
force_certified_lane = true
)");
    ASSERT_FALSE(leanByDefault.has_value());
    EXPECT_EQ(leanByDefault.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(leanByDefault.error().message.find("method.force_certified_lane"), std::string::npos);
    EXPECT_NE(leanByDefault.error().message.find("LEAN member"), std::string::npos);

    const auto explicitLean = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "lean"
accuracy = "kNormal"
force_certified_lane = true
)");
    ASSERT_FALSE(explicitLean.has_value());
    EXPECT_EQ(explicitLean.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(explicitLean.error().message.find("LEAN member"), std::string::npos);

    // The families whose own lanes do not read the request: RiEngineOptions
    // has no such field (its exchange half derives the lane from the preset),
    // and the driver does not seed QfmmOptions::useCertifiedMixedPrecision
    // (the gap recorded as an open item).
    const auto riJLink = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
aux = "def2-universal-jfit"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
force_certified_lane = true
)");
    ASSERT_FALSE(riJLink.has_value());
    EXPECT_EQ(riJLink.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(riJLink.error().message.find("ri_j_link"), std::string::npos);

    const auto qfmm = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "qfmm"
accuracy = "kNormal"
force_certified_lane = true
)");
    ASSERT_FALSE(qfmm.has_value());
    EXPECT_EQ(qfmm.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(qfmm.error().message.find("qfmm"), std::string::npos);

    // The ABSENT key is not this test's case: it is default (the probe
    // decides), and the same text without the key runs - which is what keeps a
    // rule that refused the key everywhere from passing this test.
    const auto absent = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
)");
    ASSERT_TRUE(absent.has_value()) << absent.error().message;
    EXPECT_NE(absent->find("\"converged\": true"), std::string::npos);
}

TEST(DriverErrorTest, ForcedCertifiedLaneRejectedAtKTight) {
    // The second judgement of the same resolution point, and the one the
    // sibling enforcement key does NOT share: at kTight the lane's own gate
    // admits no quartet at all - MixedPrecisionThreshold(kTight) is 0.0 by
    // construction (the strict-pins contract) - so the request cannot be
    // honoured there on ANY route, not even the machinery member. Accepting it
    // would be the silent no-op the disclosure rule forbids: the key would read as honoured
    // while the run computed on the fp64 lane. The route is explicit (the
    // machinery member, the one that CAN carry the request) so the refusal
    // below can only be the preset's.
    const auto tight = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kTight"
force_certified_lane = true
)");
    ASSERT_FALSE(tight.has_value());
    EXPECT_EQ(tight.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(tight.error().message.find("method.force_certified_lane"), std::string::npos);
    EXPECT_NE(tight.error().message.find("kTight"), std::string::npos);

    // The control leg: the SAME route and the same key at a preset whose gate
    // is open, so the refusal above is the preset's and not the route's.
    const auto normal = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
force_certified_lane = true
)");
    ASSERT_TRUE(normal.has_value()) << normal.error().message;
    EXPECT_NE(normal->find("\"converged\": true"), std::string::npos);
}

TEST(DriverErrorTest, RejectsMalformedPropertiesBlock) {
    // Unknown esp scheme names are a parse-time vocabulary error.
    const auto unknownEsp = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
[properties]
esp = "chargemodel"
)");
    ASSERT_FALSE(unknownEsp.has_value());
    EXPECT_EQ(unknownEsp.error().code, qcx::ErrorCode::kInvalidArgument);

    // Negative fragment indices are rejected by the parser.
    const auto negativeIndex = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
[properties]
nocv_fragments = [[0], [-1]]
)");
    ASSERT_FALSE(negativeIndex.has_value());
    EXPECT_EQ(negativeIndex.error().code, qcx::ErrorCode::kInvalidArgument);

    // Non-integer fragment indices are rejected by the parser.
    const auto nonIntegerIndex = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
[properties]
nocv_fragments = [[0], ["H"]]
)");
    ASSERT_FALSE(nonIntegerIndex.has_value());
    EXPECT_EQ(nonIntegerIndex.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(DriverErrorTest, RejectsOutOfRangeFragmentIndex) {
    // Index 2 does not exist in the two-atom file; ValidateCombination
    // rejects it up front, before any SCF work (the canonical mapping's
    // defensive re-check never sees it).
    const auto outOfRange = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
[properties]
nocv_fragments = [[0], [2]]
)");
    ASSERT_FALSE(outOfRange.has_value());
    EXPECT_EQ(outOfRange.error().code, qcx::ErrorCode::kInvalidArgument);
}

// The liveness analysis-charge refusal relocated into the cap-child family
// (2026-09-05): this was the one-process suite's ONLY in-process
// small-cap run - a 0.6 GiB job-object cap applied to the host test
// process (already at 1.2-2.4 GiB committed) turned every later
// thread-stack commit into a failure (the spawn-site hunt's ranked death
// candidate: bad_alloc / CreateThread 1455 / 0xC00000FD). The child owns
// the cap now, per the suite's pattern; the refusal contract is
// unchanged (the child exits 1 and the marker carries the diagnostic).
TEST(DriverErrorTest, RefusesNocvDenseTensorWhenTheCapCannotFitIt) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

#if defined(QcxHasCuda) || defined(_DEBUG)
    // CUDA build only: the fixture's child applies a 0.6 GiB cap, and a
    // CUDA-linked process cannot hold it - the runtime's load-time
    // footprint plus the device probe's init (0.6 GiB clears the
    // driver's GpuProbeFloorGiB, so the sweep legitimately runs here)
    // leave the dodecane run no headroom, and it dies with a bad_alloc
    // instead of the ladder refusal (the same environmental floor
    // as the rest of the cap-child family). The Debug leg skips the same floor: the Debug exe's
    // load-time footprint also exceeds the tiny cap (the 2026-09-08 wedge class). The refusal
    // contract is CPU-side; the CUDA-less lanes carry this fixture's coverage.
    GTEST_SKIP() << "CUDA or Debug build: the runtime's committed footprint cannot fit the 0.6 GiB "
                    "cap; the dense-tensor refusal is verified on the CUDA-less builds";
#else
    // The dodecane STO-3G SCF (n = 86, nPairs = 1953, direct base
    // 0.208 GiB) fits the 0.6 GiB cap; the dense-tensor analysis charge
    // (936,271,648 B = 0.87 GiB, the same estimate the engine's
    // admission gate applies) does not. The driver refuses with the
    // ladder BEFORE calling the builder - the child survives the 0.6 GiB
    // job-object cap and exits 1 with the graceful diagnostic in the
    // marker ("refused: <message>"), never a raw allocation failure.
    const auto marker = qcx::testing::CapChildMarkerPath(14);
    std::filesystem::remove(marker);

    const int exitCode = qcx::testing::RunCapChild(0.6, marker, "nocv_dodecane");

    EXPECT_EQ(exitCode, 1) << "the run must refuse at the analysis seam, not complete";
    const std::string contents = qcx::testing::ReadCapChildMarker(marker);
    EXPECT_NE(contents.find("refused: "), std::string::npos);
    EXPECT_NE(contents.find("dense ERI tensor"), std::string::npos);
    EXPECT_NE(contents.find("936271648"), std::string::npos);
    EXPECT_NE(contents.find("memory_cap_gib"), std::string::npos);
    EXPECT_NE(contents.find("nocv_fragments"), std::string::npos);
    std::filesystem::remove(marker);
#endif
}

// ==== molden export: the written-file mini-parser and
// the end-to-end pins. The H2 pin re-derives the electron density at the
// H nucleus from the written [GTO]/[MO] rows alone - the four-in-one
// referee: [GTO] coefficients, [MO] ordering, MO values and occupations.
// Same reimplementation convention as the io module's own tests.

namespace {

// One [GTO] shell of the written file: the label and the
// (exponent, stored coefficient) primitive rows.
struct MoldenGtoShell {
    std::string label;
    std::vector<std::pair<double, double>> primitives;
};

// One [MO] orbital: the records and the n coefficient rows
// (1-based row index -> AO slot).
struct MoldenOrbital {
    std::string spin;
    double energy = 0.0;
    double occupation = 0.0;
    std::vector<double> coefficients;
};

// The [GTO] shells in file order and the [MO] orbitals across the blocks.
struct MoldenFile {
    std::vector<MoldenGtoShell> shells;
    std::vector<MoldenOrbital> orbitals;
};

std::string ReadFileText(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

// A reader of the written [Molden Format] [Atoms] (AU) [5D] [7F] [GTO]
// [MO] file: the shell rows and the orbital records the writer emits.
MoldenFile ParseMoldenFile(const std::string& text) {
    MoldenFile file;
    std::istringstream stream(text);
    std::string line;
    bool inGto = false;
    bool inMo = false;

    while (std::getline(stream, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        if (line == "[GTO]")
        {
            inGto = true;
            continue;
        }

        if (line == "[MO]")
        {
            inGto = false;
            inMo = true;
            continue;
        }

        if (inGto)
        {
            if (line.empty())
            {
                continue;
            }

            std::istringstream row(line);
            std::string first;
            row >> first;

            if (first.empty() || (first[0] >= '0' && first[0] <= '9'))
            {
                // The first token is the exponent itself; re-parse the row
                // so it is not consumed by the lookahead.
                std::istringstream primRow(line);
                double exponent = 0.0;
                double coefficient = 0.0;
                primRow >> exponent >> coefficient;

                if (!file.shells.empty())
                {
                    file.shells.back().primitives.emplace_back(exponent, coefficient);
                }

                continue;
            }

            int primCount = 0;
            row >> primCount;

            if (primCount > 0)
            {
                file.shells.push_back(MoldenGtoShell{first, {}});
            }
        } else if (inMo)
        {
            if (line.rfind("Sym= ", 0) == 0)
            {
                file.orbitals.push_back(MoldenOrbital{});
            } else if (!line.empty() && !file.orbitals.empty())
            {
                if (line.rfind("Ene= ", 0) == 0)
                {
                    std::istringstream(line.substr(5)) >> file.orbitals.back().energy;
                } else if (line.rfind("Spin= ", 0) == 0)
                {
                    file.orbitals.back().spin = line.substr(6);
                } else if (line.rfind("Occup= ", 0) == 0)
                {
                    std::istringstream(line.substr(7)) >> file.orbitals.back().occupation;
                } else
                {
                    std::istringstream row(line);
                    int index = 0;
                    double coefficient = 0.0;
                    row >> index >> coefficient;

                    if (index > 0)
                    {
                        file.orbitals.back().coefficients.resize(static_cast<std::size_t>(index));
                        file.orbitals.back().coefficients[static_cast<std::size_t>(index - 1)] =
                            coefficient;
                    }
                }
            }
        }
    }

    return file;
}

// N_0(zeta) = (2 zeta / pi)^(3/4): the l = 0 case of the evaluator's
// per-primitive normalization (ao_evaluator.hpp:18). The written
// coefficients are used as stored - the parse-time unit-norm scale is in
// them, N_l(zeta) applies at evaluation only.
double RadialS0(double zeta, double r2) {
    return std::pow(2.0 * zeta / std::numbers::pi, 0.75) * std::exp(-zeta * r2);
}

// The TOML-basic-string-safe form of a Windows path: the backslashes are
// escapes in TOML basic strings, forward slashes are not.
std::string TomlPath(const std::filesystem::path& path) {
    std::string text = path.string();
    std::replace(text.begin(), text.end(), '\\', '/');
    return text;
}

} // namespace

TEST(DriverPinTest, H2MoldenEndToEnd) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // The H2 geometry: H at 0 and 1.4 Bohr on the x axis (kH2Toml is the
    // same fixture in Angstrom).
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "qcx_molden_h2_e2e.molden";
    std::filesystem::remove(path);

    const auto toml = std::string(kH2Toml) + "[properties]\nmolden = \"" + TomlPath(path) + "\"\n";
    auto result = RunInputText(toml.c_str());
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_NE(result->find("\"molden\""), std::string::npos);
    EXPECT_NE(result->find("\"file\""), std::string::npos);
    EXPECT_NE(result->find(TomlPath(path)), std::string::npos);
    EXPECT_NE(
        result->find("\"schema_version\": " + std::to_string(qcx::io::RunResult::kSchemaVersion)),
        std::string::npos);

    ASSERT_TRUE(std::filesystem::exists(path));
    const std::string text = ReadFileText(path);
    std::filesystem::remove(path);

    // The structural golden lines: the section headers, the [Atoms] (AU)
    // Bohr record, the STO-3G H shell line with its 1.00 column, the [MO]
    // records with the required space after '='.
    EXPECT_NE(text.find("[Molden Format]"), std::string::npos);
    EXPECT_NE(text.find("[Atoms] (AU)"), std::string::npos);
    EXPECT_NE(text.find("[5D]"), std::string::npos);
    EXPECT_NE(text.find("[7F]"), std::string::npos);
    EXPECT_NE(text.find("[GTO]"), std::string::npos);
    EXPECT_NE(text.find("[MO]"), std::string::npos);
    EXPECT_NE(text.find("H 1 1 0.000000000000 0.000000000000 0.000000000000"), std::string::npos);
    EXPECT_NE(text.find("s 3 1.00"), std::string::npos);
    EXPECT_NE(text.find("Sym= A"), std::string::npos);
    EXPECT_NE(text.find("Spin= Alpha"), std::string::npos);

    // The parse-back density pin: rho at the H nucleus recomputed from
    // the written rows alone matches the expected row 0.354892073081.
    const MoldenFile parsed = ParseMoldenFile(text);
    ASSERT_EQ(parsed.shells.size(), 2u);
    ASSERT_EQ(parsed.orbitals.size(), 2u);
    EXPECT_EQ(parsed.orbitals[0].spin, "Alpha");
    EXPECT_NEAR(parsed.orbitals[0].occupation, 2.0, 1e-12);
    EXPECT_NEAR(parsed.orbitals[1].occupation, 0.0, 1e-12);
    EXPECT_LE(parsed.orbitals[0].energy, parsed.orbitals[1].energy);

    constexpr double kHBohrX = 1.4;
    std::array<double, 2> aoAtH1{};

    for (std::size_t shell = 0; shell < parsed.shells.size(); ++shell)
    {
        const double r2 = (shell == 0) ? 0.0 : kHBohrX * kHBohrX;

        for (const auto& [exponent, coefficient] : parsed.shells[shell].primitives)
        {
            aoAtH1[shell] += coefficient * RadialS0(exponent, r2);
        }
    }

    double rho = 0.0;

    for (const auto& orbital : parsed.orbitals)
    {
        ASSERT_EQ(orbital.coefficients.size(), 2u);
        const double phi =
            orbital.coefficients[0] * aoAtH1[0] + orbital.coefficients[1] * aoAtH1[1];
        rho += orbital.occupation * phi * phi;
    }

    EXPECT_NEAR(rho, 0.354892073081, 1e-8);
}

TEST(DriverPinTest, O2MoldenUhfTwoBlocks) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // kO2Toml: the O2 STO-3G triplet SAD fixture (R = 2.2818443 Bohr on
    // the z axis). The UHF branch writes alpha then beta: nAlpha =
    // (16 + 3 - 1)/2 = 9, nBeta = 7 - one unpaired electron per oxygen.
    // The file carries the full n x n coefficient sets, so each block
    // has 10 orbitals (9 + 1 virtual, 7 + 3 virtuals).
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "qcx_molden_o2_uhf.molden";
    std::filesystem::remove(path);

    const auto toml = std::string(kO2Toml) + "[properties]\nmolden = \"" + TomlPath(path) + "\"\n";
    auto result = RunInputText(toml.c_str());
    ASSERT_TRUE(result.has_value()) << result.error().message;

    ASSERT_TRUE(std::filesystem::exists(path));
    const MoldenFile parsed = ParseMoldenFile(ReadFileText(path));
    std::filesystem::remove(path);

    EXPECT_EQ(parsed.orbitals.size(), 20u);
    ASSERT_EQ(parsed.shells.size(), 6u); // two O atoms, 1s 2s 2p each

    std::size_t alphaCount = 0;
    std::size_t betaCount = 0;
    bool betaSeen = false;
    double previousAlphaEnergy = -1e300;
    double previousBetaEnergy = -1e300;

    for (const auto& orbital : parsed.orbitals)
    {
        ASSERT_EQ(orbital.coefficients.size(), 10u);

        if (orbital.spin == "Alpha")
        {
            EXPECT_FALSE(betaSeen) << "all alpha orbitals precede the beta block";
            EXPECT_GE(orbital.energy, previousAlphaEnergy);
            previousAlphaEnergy = orbital.energy;
            EXPECT_NEAR(orbital.occupation, (alphaCount < 9) ? 1.0 : 0.0, 1e-12);
            ++alphaCount;
        } else
        {
            ASSERT_EQ(orbital.spin, "Beta");
            betaSeen = true;
            EXPECT_GE(orbital.energy, previousBetaEnergy);
            previousBetaEnergy = orbital.energy;
            EXPECT_NEAR(orbital.occupation, (betaCount < 7) ? 1.0 : 0.0, 1e-12);
            ++betaCount;
        }
    }

    EXPECT_EQ(alphaCount, 10u);
    EXPECT_EQ(betaCount, 10u);
}

TEST(DriverPinTest, MoldenAbsentWhenNotRequested) {
    // No molden key: no "molden" member in the JSON and nothing on disk.
    // The probe path doubles as the file-position check - it must not
    // exist before or after the run.
    const std::filesystem::path probe =
        std::filesystem::temp_directory_path() / "qcx_molden_absent_probe.molden";
    std::filesystem::remove(probe);

    auto result = RunInputText(kH2Toml);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(result->find("\"molden\""), std::string::npos);
    EXPECT_FALSE(std::filesystem::exists(probe));
}

// The attribution driver opt-in ([memory_instrument], attribution): the
// deferred run-level coverage test for it: an
// instrumented run must reproduce the pin exactly (the instrument is
// observational — the bit-parity pins are absolute), keep the
// coverage invariant (every tagged allocation of the run rode a term
// scope; unclassified stays empty), land its trace, and close its window
// on the way out (no joinable watchdog outlives RunDriver, so a later run
// in this process can re-enable).
TEST(DriverMemoryInstrumentTest, InstrumentedH2RunStaysOnThePinAndKeepsUnclassifiedEmpty) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const std::filesystem::path tracePath =
        std::filesystem::temp_directory_path() / "qcx_instrument_h2_trace.csv";
    std::filesystem::remove(tracePath);

    const auto toml = std::string(kH2Toml) +
                      "[memory_instrument]\nenabled = true\n"
                      "snapshot_interval_ms = 250\ntrace_file = \"" +
                      TomlPath(tracePath) + "\"\n";
    auto result = RunInputText(toml.c_str());
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto total = JsonNumber(*result, "\"total_energy_hartree\"");
    ASSERT_TRUE(total.has_value());
    EXPECT_NEAR(*total, -1.1167143252, 1e-8);

    // The run's window is closed: RunDriver disabled the instrument on its
    // way out (the guard), so the stats are final and a later run in this
    // process can re-enable.
    EXPECT_FALSE(qcx::memory::AllocationInstrumentEnabled());

    // The coverage invariant: the run's tagged traffic rode its term
    // scopes — unclassified stayed empty. The window must also have
    // captured SOMETHING: a silently inert window would make the
    // empty-unclassified claim vacuous.
    std::uint64_t taggedAllocations = 0;

    for (std::size_t tag = 0;
         tag < static_cast<std::size_t>(qcx::memory::AllocationTag::kUnclassified);
         ++tag)
    {
        taggedAllocations +=
            qcx::memory::TagAllocationCount(static_cast<qcx::memory::AllocationTag>(tag));
    }

    EXPECT_GT(taggedAllocations, 0u);
    EXPECT_EQ(qcx::memory::TagAllocationCount(qcx::memory::AllocationTag::kUnclassified), 0u);
    EXPECT_EQ(qcx::memory::TagPeakBytes(qcx::memory::AllocationTag::kUnclassified), 0u);

    // The trace: the versioned header, the run metadata (the input cap in
    // bytes — the defaulted 16.0 GiB — plus the resolved team and the tag
    // columns), and the final snapshot row, which only lands on disable.
    ASSERT_TRUE(std::filesystem::exists(tracePath)) << tracePath.string();
    const std::string trace = ReadFileText(tracePath);
    std::filesystem::remove(tracePath);

    EXPECT_NE(trace.find("# qcx allocation attribution trace v1"), std::string::npos);
    EXPECT_NE(trace.find("#meta cap_bytes = 17179869184"), std::string::npos);
    EXPECT_NE(trace.find(",final,"), std::string::npos);
}

TEST(DriverMemoryInstrumentTest, InstrumentedRunRefusesWhenTheTraceCannotOpen) {
    // Fail-closed: a trace path whose parent directory does not exist
    // cannot open, and the run refuses at entry (the enable precedes any
    // computation, so nothing heavy runs here) rather than proceeding with
    // a silently voided attribution protocol. The failed enable leaves the
    // instrument disabled — the cpp's cleanup, asserted through the public
    // query.
    const std::filesystem::path tracePath =
        std::filesystem::temp_directory_path() / "qcx_instrument_no_such_dir" / "trace.csv";

    const auto toml = std::string(kH2Toml) +
                      "[memory_instrument]\nenabled = true\n"
                      "snapshot_interval_ms = 250\ntrace_file = \"" +
                      TomlPath(tracePath) + "\"\n";
    auto result = RunInputText(toml.c_str());
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().message.find("could not be opened"), std::string::npos)
        << "message: " << result.error().message;
    EXPECT_FALSE(qcx::memory::AllocationInstrumentEnabled());
}

// The refusal record, driven end to end (
// 2026-09-13). The instrument opens at run entry, but every admission gate
// sits behind it: a refused run reaches no tagged allocation site, so before
// the fix its trace was a run's worth of watchdog rows with every per-tag
// column zero and every `predicted_high_water_*` meta value at the
// placeholder - a record that reads like a measurement instead of a
// non-event. That is the MEASURED Run A shape, not a hypothesis:
// ` is 163814 bytes / 1186 lines / 1149
// data rows (1148 tick + 1 final, 60 columns each) with all 18 family tags
// zero in every one of them, against a job peak of 11.563 GiB - the trace is
// not byte-empty, which is exactly what makes the zeroed record look like an
// answer.
//
// The refusal below is the cheapest one the driver raises AFTER the window
// opens and BEFORE any builder exists - the `enforce_certified_bound`
// resolution point, on the auto-lean default at n = 2. No SCF runs, so this
// test is the record's own test and not a memory experiment.
TEST(DriverMemoryInstrumentTest, RefusedRunRecordsTheRefusalInsteadOfAZeroedTrace) {
    const std::filesystem::path tracePath =
        std::filesystem::temp_directory_path() / "qcx_instrument_refused_trace.csv";
    std::filesystem::remove(tracePath);

    const auto toml = std::string(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
enforce_certified_bound = true
)") +
                      "[memory_instrument]\nenabled = true\n"
                      "snapshot_interval_ms = 250\ntrace_file = \"" +
                      TomlPath(tracePath) + "\"\n";

    auto result = RunInputText(toml.c_str());

    // The refusal is the driver's own (the certified-bound resolution point),
    // unchanged by the record: the funnel records, it never decides.
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(result.error().message.find("method.enforce_certified_bound"), std::string::npos)
        << "message: " << result.error().message;
    EXPECT_FALSE(qcx::memory::AllocationInstrumentEnabled());

    ASSERT_TRUE(std::filesystem::exists(tracePath)) << tracePath.string();
    const std::string trace = ReadFileText(tracePath);
    std::filesystem::remove(tracePath);

    // The window did open: the versioned header is there, which is what made
    // the pre-fix trace a plausible-looking record.
    EXPECT_NE(trace.find("# qcx allocation attribution trace v1"), std::string::npos);

    // The refusal record. The token is a line opening with `#refused ` (the
    // trailing space included); the reason is the driver's own refusal text,
    // flattened to one line. The `#` prefix keeps it out of the snapshot-row
    // grammar, so a reader that skips comments is unaffected.
    const std::size_t at = trace.find("\n#refused ");
    ASSERT_NE(at, std::string::npos) << "trace:\n" << trace;
    const std::size_t lineEnd = trace.find('\n', at + 1);
    ASSERT_NE(lineEnd, std::string::npos);
    std::string recorded = trace.substr(at + 10, lineEnd - at - 10);

    // Exactly ONE record, not one per funnel site the error passed: a run
    // returns through a single site, and the inner runners are outside the
    // funnel, so a second line would mean the error was re-wrapped somewhere.
    EXPECT_EQ(trace.find("\n#refused ", at + 1), std::string::npos) << "trace:\n" << trace;

    // The trace file is opened in text mode (CRLF on disk).
    while (!recorded.empty() && (recorded.back() == '\r' || recorded.back() == '\n'))
    {
        recorded.pop_back();
    }

    EXPECT_EQ(recorded, result.error().message)
        << "the record must carry the refusal text verbatim";
}

// The single-Fock-call timing seam (the re-scoped gate's per-call
// wall): the lean
// row family gains total_wall_ms, the CALL's own span
// (FockBuildStats::totalWallTime) as opposed to the differential reading
// of the cumulative wall column, which needs two rows and discards the
// first one anyway. The pin is the shape the harness reads: the run's
// calls are the rows of a family numbered from call=1 with no gap, the
// ONE-iteration budget's own call is its FIRST row, and that row still
// carries a per-call number - inside the cumulative wall it sits in,
// above zero. The family's LENGTH is not pinned (the body says why).
TEST(DriverRunTest, LeanStatsRowCarriesTheSingleCallWall) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const std::filesystem::path tracePath =
        std::filesystem::temp_directory_path() / "qcx_lean_call_wall_trace";
    const std::filesystem::path statsPath = tracePath.string() + ".stats";
    std::filesystem::remove(tracePath);
    std::filesystem::remove(statsPath);

    // The auto-lean H2 fixture at a ONE-pass budget with the stats
    // side-channel on: the one-call case the seam exists for. The key is
    // spliced into the fixture's own [scf] block - a second [scf] table
    // would be a duplicate-table parse error.
    std::string toml(kH2TomlAutoLean);
    const std::string budgetKey = "max_iterations = 100";
    const auto budget = toml.find(budgetKey);
    ASSERT_NE(budget, std::string::npos);
    toml.replace(budget, budgetKey.size(), "max_iterations = 1");
    const auto tolerance = toml.find("\nenergy_tolerance", budget);
    ASSERT_NE(tolerance, std::string::npos);
    toml.insert(tolerance, "\ntrace_file = \"" + TomlPath(tracePath) + "\"");

    auto result = RunInputText(toml.c_str());
    ASSERT_TRUE(result.has_value()) << result.error().message;

    ASSERT_TRUE(std::filesystem::exists(statsPath)) << statsPath.string();
    const std::string stats = ReadFileText(statsPath);
    std::filesystem::remove(statsPath);
    std::filesystem::remove(tracePath);

    // The row family's SHAPE, count-insensitively (the owner's ruling of
    // 2026-09-13): the run's Fock calls are the rows of a family numbered
    // from call=1 with no gap and no repeat, and the ONE-iteration budget's
    // own call is the FIRST row - the row the per-call reading below is
    // taken from. The family's LENGTH is deliberately not pinned: The family length
    // fix-A's exit-path recompute builds the Fock of the density the run
    // RETURNS, one trailing call per run, so a one-iteration budget emits
    // two rows where this test used to read one, and any pinned length
    // would have to move again for the next by-design build. The numbering
    // still catches a stream that does not start at the run's first call
    // and a duplicated, skipped or reordered row.
    const auto callIndices = [&stats]() {
        std::vector<long> indices;
        std::istringstream lines(stats);
        std::string line;

        // The harness's own row contract: a parseable row is a line carrying
        // both a "call=" and a "wall=" token.
        while (std::getline(lines, line))
        {
            const auto at = line.find("call=");

            if (at == std::string::npos || line.find("wall=") == std::string::npos)
            {
                continue;
            }

            try
            { indices.push_back(std::stol(line.substr(at + 5))); } catch (const std::exception&)
            { indices.push_back(-1); }
        }

        return indices;
    }();
    ASSERT_FALSE(callIndices.empty()) << "the seam recorded no per-call rows: " << stats;

    for (std::size_t i = 0; i < callIndices.size(); ++i)
    {
        EXPECT_EQ(callIndices[i], static_cast<long>(i) + 1)
            << "the row family is not the run's calls in order: " << stats;
    }

    const auto column = [&stats](const std::string& key) -> std::optional<double> {
        const auto at = stats.find(key);

        if (at == std::string::npos)
        {
            return std::nullopt;
        }

        return std::stod(stats.substr(at + key.size()));
    };
    const auto totalMillis = column("total_wall_ms=");
    ASSERT_TRUE(totalMillis.has_value()) << stats;
    const auto cumulativeSeconds = column("wall=");
    ASSERT_TRUE(cumulativeSeconds.has_value()) << stats;

    // A lean call does real work even at two basis functions, and the call
    // sits inside the stream's life - the cumulative wall column starts
    // before the builder Create and the guess, so it can only read at or
    // above the call whose span it carries.
    EXPECT_GT(*totalMillis, 0.0);
    EXPECT_LE(*totalMillis, *cumulativeSeconds * 1000.0);
}

// The trace-wiring contract pin (the silent-unmeasurable-builder defect of
// the defaults tournament, window-20260911-063141Z): a builder family whose
// seam records per-call rows must produce them. A traced run that exits 0
// with an EMPTY <trace_file>.stats is indistinguishable from a wiring
// defect to any harness that counts rows - which is exactly how the qfmm
// cells came out ERROR: the RHF kQfmm branch of WireRhfFockBuilder took the
// generic no-stats seam, its null sink left the trace file holding zero
// rows, and no timing limb could ever measure (the harness's
// no_scf_trace_rows abort, exit 0 and an empty file either way).
//
// This test runs the RHF families the fixtures can back - the lean member
// (the absent key), the machinery direct member, and the composed QFMM
// family - with a trace requested, and asserts each lands parseable rows.
// A family that loses its sink fails HERE, by name, instead of in a
// tournament window. The ri_j_link family is not covered: the sto-3g
// fixture can supply no aux basis (the family's documented preflight
// refusal), so it needs a fixture of its own.
//
// The QFMM arm is also the honesty pin on its own row family: the composed
// builder exposes no FockBuildStats at all (its Coulomb half's far-field
// multipole passes carry no quartet count), so its rows carry the call's
// wall and the far-field liveness and NONE of the quartet counters - a
// fabricated nq64=0/np=0/g=0 row would pass a bare row-count check while
// telling a lie, so the absence is asserted, not just the row.
TEST(DriverRunTest, TraceEnabledRunsEmitTheRowFamilyOfTheirBuilder) {
    // The harness's own row contract (tournament.py ParseStatsRows): a
    // parseable row is a line carrying both a "call=" and a "wall=" token.
    // Anything else is not a measurement, however many lines the file has.
    const auto countRows = [](const std::string& stats) {
        std::size_t rows = 0;
        std::istringstream lines(stats);
        std::string line;

        while (std::getline(lines, line))
        {
            if (line.find("call=") != std::string::npos && line.find("wall=") != std::string::npos)
            {
                ++rows;
            }
        }

        return rows;
    };

    // One traced run: the [scf] trace_file key is spliced into the
    // fixture's own block (a second [scf] table would be a duplicate-table
    // parse error) and the run's .stats text comes back.
    const auto tracedStats = [](std::string toml, const std::string& stem) {
        const std::filesystem::path tracePath = std::filesystem::temp_directory_path() / stem;
        const std::filesystem::path statsPath = tracePath.string() + ".stats";
        std::filesystem::remove(statsPath);
        std::filesystem::remove(tracePath);

        const auto tolerance = toml.find("\nenergy_tolerance");
        EXPECT_NE(tolerance, std::string::npos);
        toml.insert(tolerance, "\ntrace_file = \"" + TomlPath(tracePath) + "\"");

        auto result = RunInputText(toml.c_str());
        EXPECT_TRUE(result.has_value()) << result.error().message;

        const std::string stats =
            std::filesystem::exists(statsPath) ? ReadFileText(statsPath) : std::string();
        std::filesystem::remove(statsPath);
        std::filesystem::remove(tracePath);
        return stats;
    };

    // The lean member (the absent fock_builder key): the LEAN row family -
    // the screened-quartet counter and the call's own wall span.
    const std::string leanStats = tracedStats(kH2TomlAutoLean, "qcx_trace_lean_rows");
    EXPECT_GE(countRows(leanStats), 2u)
        << "the lean seam recorded no per-call rows; stats: " << leanStats;
    EXPECT_NE(leanStats.find("total_wall_ms="), std::string::npos) << leanStats;
    // The kernel-span probe (the kernel-span instrument): the lean row carries the
    // three kernel sub-spans and their four counters in the SAME row family,
    // never a parallel stream. The columns are presence-driven like the rest
    // of the tail, so their absence is the failure this asserts.
    for (const std::string_view column : {"eri_wall_ms=",
                                          "eri_prep_wall_ms=",
                                          "contract_wall_ms=",
                                          "kernel_vrr_wall_ms=",
                                          "kernel_ket_wall_ms=",
                                          "kernel_bra_wall_ms=",
                                          "kernel_vrr_quads=",
                                          "kernel_gate_seam_calls=",
                                          "kernel_groups=",
                                          "kernel_prim_passes="})
    {
        EXPECT_NE(leanStats.find(column), std::string::npos)
            << "the lean row is missing the \"" << column << "\" column: " << leanStats;
    }

    // The decomposition's own arithmetic, read off the row the way the
    // harness reads it: the three kernel sub-spans never sum past the kernel
    // span they subdivide (eri - prep). The check is an inequality - the
    // per-batch layout walk, scratch.assign, the group splitting and the
    // loop overheads belong to no phase - and it is asserted at the row
    // level so a column that stops being written, or two that start
    // overlapping, fails HERE rather than in a reader's arithmetic.
    const auto statsColumnOf = [](const std::string& stats,
                                  const std::string& key) -> std::optional<double> {
        const auto at = stats.find(key);

        if (at == std::string::npos)
        {
            return std::nullopt;
        }

        try
        { return std::stod(stats.substr(at + key.size())); } catch (const std::exception&)
        { return std::nullopt; }
    };
    const auto statsColumn = [&leanStats](const std::string& key) -> std::optional<double> {
        const auto at = leanStats.find(key);

        if (at == std::string::npos)
        {
            return std::nullopt;
        }

        try
        { return std::stod(leanStats.substr(at + key.size())); } catch (const std::exception&)
        { return std::nullopt; }
    };
    const auto eriMilliseconds = statsColumn("eri_wall_ms=");
    const auto prepMilliseconds = statsColumn("eri_prep_wall_ms=");
    const auto vrrMilliseconds = statsColumn("kernel_vrr_wall_ms=");
    const auto ketMilliseconds = statsColumn("kernel_ket_wall_ms=");
    const auto braMilliseconds = statsColumn("kernel_bra_wall_ms=");

    ASSERT_TRUE(eriMilliseconds.has_value()) << leanStats;
    ASSERT_TRUE(prepMilliseconds.has_value()) << leanStats;
    ASSERT_TRUE(vrrMilliseconds.has_value()) << leanStats;
    ASSERT_TRUE(ketMilliseconds.has_value()) << leanStats;
    ASSERT_TRUE(braMilliseconds.has_value()) << leanStats;
    EXPECT_GE(*vrrMilliseconds + *ketMilliseconds + *braMilliseconds,
              0.0); // a row whose phases read negative is not a span set
    EXPECT_LE(*vrrMilliseconds + *ketMilliseconds + *braMilliseconds,
              *eriMilliseconds - *prepMilliseconds)
        << "the kernel sub-spans exceeded eri - prep: " << leanStats;

    // The machinery direct member: the machinery row family - the quartet
    // counters, the calibration terms AND the call's own wall span.
    const std::string directStats = tracedStats(kH2Toml, "qcx_trace_direct_rows");
    EXPECT_GE(countRows(directStats), 2u)
        << "the direct seam recorded no per-call rows; stats: " << directStats;
    EXPECT_NE(directStats.find("nq64="), std::string::npos) << directStats;
    // The seam-measured span (see the row-family comment in run_driver.cpp):
    // read as a NUMBER, not only as a token, because a column that printed a
    // fabricated zero would pass a presence check. The one-call timing rows
    // read their call wall from exactly this column.
    EXPECT_NE(directStats.find("total_wall_ms="), std::string::npos) << directStats;
    const auto directWallMs = statsColumnOf(directStats, "total_wall_ms=");
    ASSERT_TRUE(directWallMs.has_value()) << directStats;
    EXPECT_GT(*directWallMs, 0.0) << directStats;

    // The RI-J link leg of the SAME machinery row family: its BuildFock takes
    // the two-argument form, so the seam hands it the stats out-parameter
    // without a bound pointer, and it forwards that parameter to its nested
    // exchange-only direct half. The builder-backed totalWallTime on this leg
    // is therefore the NESTED half's span rather than the call's - which is
    // exactly why the recorded column is the SEAM's own span (see the
    // row-family comment in run_driver.cpp). This leg is what pins that:
    // a future reader who "simplified" the column back to stats.totalWallTime
    // would publish the half's number here under the call's name.
    if (qcx::integrals::SupportsL(3))
    {
        // Its own staging variant: this fixture carries no [scf] table to
        // splice a trace_file line into, so a whole block is appended.
        const auto tracedStatsWithScf = [](std::string toml, const std::string& stem) {
            const std::filesystem::path tracePath = std::filesystem::temp_directory_path() / stem;
            const std::filesystem::path statsPath = tracePath.string() + ".stats";
            std::filesystem::remove(statsPath);
            std::filesystem::remove(tracePath);

            toml += "\n[scf]\nmax_iterations = 8\nenergy_tolerance = 1e-08\n"
                    "density_tolerance = 1e-06\ntrace_file = \"" +
                    TomlPath(tracePath) + "\"\n";

            const auto result = RunInputText(toml.c_str());
            EXPECT_TRUE(result.has_value()) << result.error().message;

            const std::string stats =
                std::filesystem::exists(statsPath) ? ReadFileText(statsPath) : std::string();
            std::filesystem::remove(statsPath);
            std::filesystem::remove(tracePath);
            return stats;
        };

        const std::string riJStats =
            tracedStatsWithScf(H2oDef2SvpToml("ri_j_link"), "qcx_trace_ri_j_rows");
        EXPECT_GE(countRows(riJStats), 2u)
            << "the RI-J link recorded no per-call rows; stats: " << riJStats;
        EXPECT_NE(riJStats.find("nq64="), std::string::npos) << riJStats;
        const auto riJWallMs = statsColumnOf(riJStats, "total_wall_ms=");
        ASSERT_TRUE(riJWallMs.has_value()) << riJStats;
        EXPECT_GT(*riJWallMs, 0.0) << riJStats;

        // The SAME family on the unrestricted leg (the per-spin adapter this
        // wiring landed): one SCF iteration makes THREE calls - the J-only half
        // and one exchange half per spin - so the rows come in threes, the
        // J call's quartet count is a TRUE zero (the RI contractions
        // evaluate no quartets, BuildCoulombOnly writes its structural zero
        // rather than leaving the sink stale) and each exchange half sizes
        // xvol with its own channel's occupancy. A wiring that resolved
        // ri_j_link and ran the direct family would record rows too - and
        // would fail the sibling pin's tensor_bytes measurement - so this
        // leg's subject is the shape of the RI-J link's own rows.
        const std::string uhfRiJStats =
            tracedStatsWithScf(H2oDef2SvpUhfToml("ri_j_link"), "qcx_trace_uhf_ri_j_rows");
        EXPECT_GE(countRows(uhfRiJStats), 3u)
            << "the unrestricted RI-J link recorded less than one iteration's three rows; stats: "
            << uhfRiJStats;
        EXPECT_NE(uhfRiJStats.find("total_wall_ms="), std::string::npos) << uhfRiJStats;
        EXPECT_NE(uhfRiJStats.find("nq64=0 "), std::string::npos)
            << "the J-only half's row is missing (its quartet count is a true zero, not an "
               "absent column); stats: "
            << uhfRiJStats;
        EXPECT_NE(uhfRiJStats.find("xvol=0 "), std::string::npos) << uhfRiJStats;

        // The SAME row family on the composed full-RI leg (the per-spin
        // adapter landed here): its call contracts BOTH halves at once,
        // so ONE SCF iteration makes TWO calls - one per spin - and each
        // row's quartet count is a TRUE zero (this builder calls no quartet
        // kernel on any path, and both of its halves come from the one
        // traversal the row times). The occupancy column is therefore the
        // only column that could differ from its sibling's and it cannot:
        // xvol is significantQuartets * occupancy and significantQuartets is
        // 0 here, which is exactly why the seam's choice of WHICH occupancy a
        // combined J-and-K call reports (this spin's exchange count, not 0 and
        // not a blend) is a labelling decision rather than a numeric one. The
        // row COUNT is what this leg pins: a wiring that dropped the seam, or
        // ran a family that builds both halves in one call, cannot land two
        // rows per iteration.
        const std::string uhfRiJkStats =
            tracedStatsWithScf(H2oDef2SvpUhfToml("ri_jk"), "qcx_trace_uhf_ri_jk_rows");
        EXPECT_GE(countRows(uhfRiJkStats), 2u)
            << "the unrestricted composed-full-RI seam recorded less than one iteration's two "
               "rows (one per spin); stats: "
            << uhfRiJkStats;
        EXPECT_NE(uhfRiJkStats.find("total_wall_ms="), std::string::npos) << uhfRiJkStats;
        EXPECT_NE(uhfRiJkStats.find("nq64=0 "), std::string::npos)
            << "the pair's row is missing its structural quartet zero; stats: " << uhfRiJkStats;
        EXPECT_NE(uhfRiJkStats.find("xvol=0 "), std::string::npos) << uhfRiJkStats;
    }

    // The UHF lean seam (the unrestricted seam): the SAME lean row family on the
    // per-spin leg, where one SCF iteration makes THREE build calls - the
    // J-only half and the two exchange halves - so the rows come in threes.
    // The J-only call's xvol is a true zero (it contracts no exchange) and
    // the exchange calls size theirs with the per-spin K occupancy, which
    // is the composed-QFMM-UHF arm's honesty problem solved the other way
    // round: the lean half-mode builders count quartets, so the UHF row
    // carries real numbers instead of absent columns.
    const std::string uhfLeanStats = tracedStats(kO2AutoLeanToml, "qcx_trace_uhf_lean_rows");
    EXPECT_GE(countRows(uhfLeanStats), 3u)
        << "the UHF lean seam recorded no per-call rows; stats: " << uhfLeanStats;
    EXPECT_NE(uhfLeanStats.find("total_wall_ms="), std::string::npos) << uhfLeanStats;
    EXPECT_NE(uhfLeanStats.find("nq64="), std::string::npos) << uhfLeanStats;
    EXPECT_NE(uhfLeanStats.find("xvol=0 "), std::string::npos) << uhfLeanStats;

    // The composed QFMM family: the QFMM row family (see the test note) -
    // the call's wall span plus the far-field liveness, and none of the
    // quartet counters the composed builder cannot back.
    const std::string qfmmStats =
        tracedStats(WithBuilderAndAccuracy(kH2Toml, "qfmm", "kTight"), "qcx_trace_qfmm_rows");
    EXPECT_GE(countRows(qfmmStats), 2u)
        << "the composed QFMM seam recorded no per-call rows - the qfmm "
           "cells cannot measure; stats: "
        << qfmmStats;
    EXPECT_NE(qfmmStats.find("total_wall_ms="), std::string::npos) << qfmmStats;
    EXPECT_NE(qfmmStats.find("far_pairs="), std::string::npos) << qfmmStats;

    // The honesty pin: the far field has no quartet representation, so the
    // composed builder reports none. Every one of these columns would be a
    // false zero if the seam took the generic no-stats path and recorded a
    // default-constructed FockBuildStats - the failure mode this test
    // exists to catch, and one a row COUNT alone cannot see.
    for (const std::string_view absent : {"nq64=", "nq32=", "np=", "g=", "xvol=", "merge=", "fpb="})
    {
        EXPECT_EQ(qfmmStats.find(absent), std::string::npos)
            << "the composed QFMM row fabricated a \"" << absent << "\" column: " << qfmmStats;
    }
}

// ---------------------------------------------------------------------------
// The restart run-restart acceptance (the restart feature): the driver
// writes the last-iterate checkpoint at the budget exit (scf.
// checkpoint_file), a later run seeds from it (guess restart), and the
// resumed trajectory is the one-shot's tail. The storage suite proves the
// file-level Save/Load round trip bit-identically; these tests prove the
// driver path - TOML -> RunInput -> checkpoint write at the exit, and
// file -> LoadScfCheckpoint -> RhfOptions seed -> solver.

namespace {

// The binary per-iteration dump reader of the SCF side-channel (the
// ScfDensityDumpWriter spec, scf/src/internal/scf_common.hpp - a private
// implementation header, so the reader is re-derived here the way the
// Python consumer tools/amf_density_invariants.py does): the 8-byte magic
// "QXCDFDMP", a little-endian u64 dimension, then records of u8 tag + LE
// u64 iteration + LE u64 dimension + dimension*dimension LE doubles in
// row-major order. x86-64 is little-endian, so the LE integers decode
// natively. Returns the tag-2 (density) records in file order, each as
// (iteration, n*n row-major doubles). The writer emits one record per
// loop pass, the density IN HAND before any DIIS extrapolation or Fock
// build - so the iteration-0 record of a restarted run IS the seeded
// density.
struct DensityDumpRecord {
    std::uint64_t iteration;
    std::vector<double> rowMajor;
};

std::optional<std::vector<DensityDumpRecord>> ReadDensityDumpRecords(
    const std::filesystem::path& path) {
    std::ifstream stream(path.string(), std::ios::binary);

    if (!stream.is_open())
    {
        return std::nullopt;
    }

    char magic[8] = {};

    if (!stream.read(magic, 8) || std::memcmp(magic, "QXCDFDMP", 8) != 0)
    {
        return std::nullopt;
    }

    std::uint64_t n = 0;

    if (!stream.read(reinterpret_cast<char*>(&n), sizeof(n)))
    {
        return std::nullopt;
    }

    std::vector<DensityDumpRecord> records;

    for (;;)
    {
        std::uint8_t tag = 0;
        std::uint64_t iteration = 0;
        std::uint64_t dimension = 0;

        if (!stream.read(reinterpret_cast<char*>(&tag), sizeof(tag)))
        {
            break; // Clean end of stream.
        }

        if (!stream.read(reinterpret_cast<char*>(&iteration), sizeof(iteration)) ||
            !stream.read(reinterpret_cast<char*>(&dimension), sizeof(dimension)) ||
            dimension != n || dimension == 0 || dimension > 10000)
        {
            return std::nullopt;
        }

        std::vector<double> matrix(static_cast<std::size_t>(dimension * dimension));

        if (!stream.read(reinterpret_cast<char*>(matrix.data()),
                         static_cast<std::streamsize>(matrix.size() * sizeof(double))))
        {
            return std::nullopt;
        }

        if (tag == 2)
        {
            records.push_back(DensityDumpRecord{iteration, std::move(matrix)});
        }
    }

    return records;
}

// Bit-exact equality between an Eigen matrix (column-major storage) and
// one row-major dump record: the refilled matrix must be byte-identical -
// a trajectory-replication claim is absolute (0.0 vs -0.0 differ in
// bytes, and that difference matters downstream).
bool SameDensityBytes(const Eigen::MatrixXd& matrix, const std::vector<double>& rowMajor) {
    if (matrix.rows() != matrix.cols() ||
        rowMajor.size() != static_cast<std::size_t>(matrix.rows() * matrix.cols()))
    {
        return false;
    }

    Eigen::MatrixXd refilled(matrix.rows(), matrix.cols());

    for (Eigen::Index row = 0; row < matrix.rows(); ++row)
    {
        for (Eigen::Index col = 0; col < matrix.cols(); ++col)
        {
            refilled(row, col) = rowMajor[static_cast<std::size_t>(row * matrix.cols() + col)];
        }
    }

    return std::memcmp(refilled.data(), matrix.data(), rowMajor.size() * sizeof(double)) == 0;
}

// The molecule the driver builds from kH2oToml, replicated bit-exactly:
// the same Angstrom literals times kAngstromToBohr (MakeCoordinates in
// run_driver.cpp) through the same Molecule::Create. The restart
// checkpoint fingerprint hashes the canonical molecule's coordinates
// byte-wise (storage/fingerprint.cpp), so a load cross-check needs this
// replica - the h2o_sto3g.hpp fixture's Bohr values are only numerically
// equal to the driver's round trip, not byte-equal.
qcx::Result<qcx::molecule::Molecule> H2oMoleculeLikeTheDriverBuildsIt() {
    std::vector<qcx::molecule::Atom> atoms{{"O", 8, 0.0}, {"H", 1, 0.0}, {"H", 1, 0.0}};
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 0.7569503270127429 * qcx::molecule::kAngstromToBohr;
    (*coordinates)(1, 1) = 0.58588227657553 * qcx::molecule::kAngstromToBohr;
    (*coordinates)(1, 2) = 0.0;
    (*coordinates)(2, 0) = -0.7569503270127429 * qcx::molecule::kAngstromToBohr;
    (*coordinates)(2, 1) = 0.58588227657553 * qcx::molecule::kAngstromToBohr;
    (*coordinates)(2, 2) = 0.0;
    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

// The H2O text of one restart-test leg: kH2oToml (no [scf]/[guess]
// blocks) plus the tight explicit [scf] gate at the caller's budget and
// extra keys (checkpoint_file / density_dump lines).
std::string H2oScfLegToml(int maxIterations, const std::string& scfExtras) {
    return std::string(kH2oToml) + "[scf]\nmax_iterations = " + std::to_string(maxIterations) +
           "\n"
           // Measured 2026-09-16 and VESTIGIAL: the H2O/STO-3G direct walk gives
           // -74.96292827079816 at the operating default in 8 iterations against the
           // tight pair's -74.96292827079819 in 10 - a 3e-14 separation. The restart
           // cells assert this walk at 1e-5 (EXPECT_NEAR(*resumedTotal,
           // -74.96292827, 1e-5)), five orders looser than either value sits from
           // the pin, and their partial+resumed==one-shot identity shares the gate
           // between both legs, so it cancels.
           "energy_tolerance = 1e-8\ndensity_tolerance = 1e-6\n"
           "use_diis = true\n" +
           scfExtras;
}

// The resumed leg: the same SCF gate at the full budget, with guess
// restart naming the checkpoint file and the caller's [scf] extras (the
// resumed leg's own density_dump, when the seeded start must be
// observed).
std::string H2oRestartLegToml(const std::filesystem::path& checkpoint,
                              const std::string& scfExtras) {
    return H2oScfLegToml(100, scfExtras) + "[guess]\ntype = \"restart\"\nrestart_path = \"" +
           TomlPath(checkpoint) + "\"\n";
}

} // namespace

TEST(DriverRestartTest, BudgetExitCheckpointSeedsTheResumedRun) {
#if !defined(QcxHasStorage)
    // Both legs of this case go through the checkpoint store - the partial run
    // writes it (scf.checkpoint_file) and the resumed run reads it (guess
    // restart) - and that store is the storage module's, which a build
    // configured with QCX_ENABLE_IO=OFF does not compile. The driver refuses
    // the write by name there, so the case has nothing to assert.
    GTEST_SKIP() << "storage module not built (QCX_ENABLE_IO=OFF): this case writes and reads the "
                    "checkpoint store";
#else
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // The partial run: H2O at the tight gate, capped at three SCF passes -
    // nowhere near the count the DIIS run needs, so the run exits NOT
    // converged and the driver writes the last-iterate checkpoint
    // (scf.checkpoint_file). A stale file pre-exists at the path: the
    // store is append-only by contract, so the driver must remove it
    // first - a leftover would otherwise refuse the write (the
    // gate-③ rerun case means the path carries a previous attempt).
    const std::filesystem::path checkpoint =
        std::filesystem::temp_directory_path() / "qcx_restart_h2o_partial.h5";
    const std::filesystem::path partialDump =
        std::filesystem::temp_directory_path() / "qcx_restart_h2o_partial.dmp";
    const std::filesystem::path resumedDump =
        std::filesystem::temp_directory_path() / "qcx_restart_h2o_resumed.dmp";
    std::filesystem::remove(checkpoint);
    std::filesystem::remove(partialDump);
    std::filesystem::remove(resumedDump);
    {
        std::ofstream stale(checkpoint, std::ios::binary);
        stale << "stale garbage from a previous attempt";
    }

    const std::string partialToml =
        H2oScfLegToml(3,
                      "checkpoint_file = \"" + TomlPath(checkpoint) + "\"\ndensity_dump = \"" +
                          TomlPath(partialDump) + "\"\n");
    auto partial = RunInputText(partialToml.c_str());
    ASSERT_TRUE(partial.has_value()) << partial.error().message;
    EXPECT_NE(partial->find("\"converged\": false"), std::string::npos);
    const auto partialIterations = JsonNumber(*partial, "\"iterations\"");
    ASSERT_TRUE(partialIterations.has_value());
    EXPECT_EQ(*partialIterations, 3.0);

    // The stale file was replaced by a real checkpoint store (the
    // remove-first write; a junk HDF5 file would fail the load below).
    ASSERT_TRUE(std::filesystem::exists(checkpoint));

    // The resumed run: the same molecule/basis/method at the same gate,
    // with guess restart naming the checkpoint. It must converge to the
    // H2O pin, and its iteration count plus the partial's three must
    // equal the one-shot's: the seed (density + DIIS history + previous
    // energies) makes the resume the one-shot's tail, not a fresh walk.
    // The resumed leg carries its own density_dump so the seeded start
    // (the iteration-0 record, the seed in hand before the first Fock
    // build) can be compared byte-for-byte against the stored density.
    auto resumed = RunInputText(
        H2oRestartLegToml(checkpoint, "density_dump = \"" + TomlPath(resumedDump) + "\"\n")
            .c_str());
    ASSERT_TRUE(resumed.has_value()) << resumed.error().message;
    EXPECT_NE(resumed->find("\"converged\": true"), std::string::npos);
    const auto resumedTotal = JsonNumber(*resumed, "\"total_energy_hartree\"");
    ASSERT_TRUE(resumedTotal.has_value());
    EXPECT_NEAR(*resumedTotal, -74.96292827, 1e-5);

    auto oneShot = RunInputText(H2oScfLegToml(100, "").c_str());
    ASSERT_TRUE(oneShot.has_value()) << oneShot.error().message;
    EXPECT_NE(oneShot->find("\"converged\": true"), std::string::npos);
    const auto oneShotIterations = JsonNumber(*oneShot, "\"iterations\"");
    ASSERT_TRUE(oneShotIterations.has_value());
    const auto resumedIterations = JsonNumber(*resumed, "\"iterations\"");
    ASSERT_TRUE(resumedIterations.has_value());
    EXPECT_EQ(*partialIterations + *resumedIterations, *oneShotIterations);

    // The stored state cross-checked through the storage reader (the
    // same call the driver's restart read made, with a molecule the
    // driver built): the checkpoint carries the RHF channel (density,
    // DIIS history, gate energies - the "DIIS history is carried"
    // evidence, since the iteration equality above holds only when the
    // resume re-ran with the restored history).
    auto molecule = H2oMoleculeLikeTheDriverBuildsIt();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto state = qcx::storage::LoadScfCheckpoint(checkpoint, *molecule, "sto-3g", "");
    ASSERT_TRUE(state.has_value()) << state.error().message;
    EXPECT_EQ(state->density.rows(), 7); // H2O/STO-3G: 5 O functions + 1 + 1.
    EXPECT_TRUE(state->densityAlpha.size() == 0);
    EXPECT_TRUE(state->densityBeta.size() == 0);
    EXPECT_FALSE(state->diis.fockHistory.empty());
    EXPECT_FALSE(state->diis.errorHistory.empty());
    EXPECT_NE(state->previousTotalEnergy, 0.0);
    EXPECT_NE(state->previousElectronicEnergy, 0.0);

    // The seeded start: the resumed run's first density record - the
    // iteration-0 pass writes the seed in hand BEFORE the first Fock
    // build - is byte-identical to the density the checkpoint stores
    // ("the restart run's first-iteration density equals the saved one").
    const auto resumedRecords = ReadDensityDumpRecords(resumedDump);
    const auto partialRecords = ReadDensityDumpRecords(partialDump);
    ASSERT_TRUE(resumedRecords.has_value());
    ASSERT_TRUE(partialRecords.has_value());
    ASSERT_FALSE(resumedRecords->empty());
    ASSERT_FALSE(partialRecords->empty());
    EXPECT_EQ(resumedRecords->front().iteration, 0u);
    EXPECT_EQ(partialRecords->back().iteration, 2u); // The budget-3 exit pass.
    EXPECT_TRUE(SameDensityBytes(state->density, resumedRecords->front().rowMajor));

    std::filesystem::remove(checkpoint);
    std::filesystem::remove(partialDump);
    std::filesystem::remove(resumedDump);
#endif
}

TEST(DriverRestartTest, ConvergedExitWritesTheCheckpointOnlyWhenAsked) {
#if !defined(QcxHasStorage)
    // This case writes the checkpoint store through scf.checkpoint_file and
    // scf.checkpoint_converged, and that store is the storage module's, which a
    // build configured with QCX_ENABLE_IO=OFF does not compile. The driver
    // refuses the write by name there, so the case has nothing to assert.
    GTEST_SKIP() << "storage module not built (QCX_ENABLE_IO=OFF): this case writes the "
                    "checkpoint store";
#else
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // kH2Toml already carries its [scf] block, so the checkpoint keys are
    // spliced in after its last key (a second [scf] header would be a
    // duplicate-table TOML error). H2 converges in a handful of passes,
    // so both legs are cheap.
    const auto withCheckpointKeys = [](const std::filesystem::path& path, bool alsoOnConvergence) {
        constexpr std::string_view kUseDiisToken = "use_diis = true";
        std::string text = kH2Toml;
        const auto pos = text.find(kUseDiisToken);
        EXPECT_NE(pos, std::string::npos);
        std::string keys = "checkpoint_file = \"" + TomlPath(path) + "\"\n";

        if (alsoOnConvergence)
        {
            keys += "checkpoint_converged = true\n";
        }

        if (pos != std::string::npos)
        {
            text.replace(pos, kUseDiisToken.size(), std::string(kUseDiisToken) + "\n" + keys);
        }

        return text;
    };

    // The converged exit writes nothing by default: checkpoint_file
    // alone targets the budget-exit case, and a converged run has no
    // last-iterate worth resuming from (the file must not appear).
    const std::filesystem::path notWritten =
        std::filesystem::temp_directory_path() / "qcx_restart_h2_converged.h5";
    std::filesystem::remove(notWritten);

    auto noWrite = RunInputText(withCheckpointKeys(notWritten, false).c_str());
    ASSERT_TRUE(noWrite.has_value()) << noWrite.error().message;
    EXPECT_NE(noWrite->find("\"converged\": true"), std::string::npos);
    EXPECT_FALSE(std::filesystem::exists(notWritten));

    // checkpoint_converged = true widens the write to the converged
    // exit: the same converged run now leaves a checkpoint behind.
    const std::filesystem::path alsoWritten =
        std::filesystem::temp_directory_path() / "qcx_restart_h2_converged_always.h5";
    std::filesystem::remove(alsoWritten);

    auto written = RunInputText(withCheckpointKeys(alsoWritten, true).c_str());
    ASSERT_TRUE(written.has_value()) << written.error().message;
    EXPECT_NE(written->find("\"converged\": true"), std::string::npos);
    EXPECT_TRUE(std::filesystem::exists(alsoWritten));
    EXPECT_GT(std::filesystem::file_size(alsoWritten), 0u);

    std::filesystem::remove(alsoWritten);
#endif
}

TEST(DriverErrorTest, RestartAndCheckpointCombinationRejections) {
    // The restart read and the checkpoint write are RHF-only in v1: on a
    // UHF run neither is silently ignored - the read has no UHF solver to
    // seed and the write would produce a file nothing could consume.
    const auto uhfRestart = RunInputText(R"(
[molecule]
charge = 1
multiplicity = 2
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"
[guess]
type = "restart"
restart_path = "anything.h5"
)");
    ASSERT_FALSE(uhfRestart.has_value());
    EXPECT_EQ(uhfRestart.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(uhfRestart.error().message.find("guess restart"), std::string::npos);

    const auto uhfCheckpoint = RunInputText(R"(
[molecule]
charge = 1
multiplicity = 2
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"
[scf]
checkpoint_file = "anything.h5"
)");
    ASSERT_FALSE(uhfCheckpoint.has_value());
    EXPECT_EQ(uhfCheckpoint.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(uhfCheckpoint.error().message.find("scf.checkpoint_file"), std::string::npos);

#if !defined(QcxHasStorage)
    // The two legs above are input validation and hold on every build: the UHF
    // restart read and the UHF checkpoint write are refused before any storage
    // call. Everything below goes through the checkpoint store, which a build
    // configured with QCX_ENABLE_IO=OFF does not compile - there the driver
    // refuses each of these reads with the build's own cause, which names
    // neither guess.restart_path nor a UHF channel.
    GTEST_SKIP() << "storage module not built (QCX_ENABLE_IO=OFF): the legs below read the "
                    "checkpoint store";
#else
    // A missing store and a store that is not a checkpoint at all both
    // fail the restart read with the schema key named; a junk file
    // exercises the driver's load-error propagation (the storage error's
    // own text travels behind the prefix).
    const auto missing = RunInputText(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
[guess]
type = "restart"
restart_path = "qcx_no_such_checkpoint_anywhere.h5"
)");
    ASSERT_FALSE(missing.has_value());
    EXPECT_NE(missing.error().message.find("guess.restart_path"), std::string::npos);

    const std::filesystem::path junk =
        std::filesystem::temp_directory_path() / "qcx_restart_junk.h5";
    {
        std::ofstream stream(junk, std::ios::binary);
        stream << "this is not an HDF5 store";
    }

    const auto junkToml = H2oRestartLegToml(junk, "");
    auto junkRun = RunInputText(junkToml.c_str());
    ASSERT_FALSE(junkRun.has_value());
    EXPECT_NE(junkRun.error().message.find("guess.restart_path"), std::string::npos);
    std::filesystem::remove(junk);

    // A checkpoint that stores a UHF state (only storage-module clients
    // can produce one in v1 - the driver refuses to write it) must be
    // refused by name on the RHF restart path: without the driver's
    // channel check the empty RHF density would silently fall back to the
    // default GWH start, i.e. a restart that is not a restart.
    auto molecule = H2oMoleculeLikeTheDriverBuildsIt();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const std::filesystem::path uhfStore =
        std::filesystem::temp_directory_path() / "qcx_restart_uhf_store.h5";
    std::filesystem::remove(uhfStore);
    qcx::scf::ScfRestartState uhfState;
    uhfState.densityAlpha = Eigen::MatrixXd::Zero(7, 7);
    uhfState.densityBeta = Eigen::MatrixXd::Zero(7, 7);
    auto saved = qcx::storage::SaveScfCheckpoint(uhfStore, *molecule, "sto-3g", "", uhfState);
    ASSERT_TRUE(saved.has_value()) << saved.error().message;

    auto uhfIntoRhf = RunInputText(H2oRestartLegToml(uhfStore, "").c_str());
    ASSERT_FALSE(uhfIntoRhf.has_value());
    EXPECT_EQ(uhfIntoRhf.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(uhfIntoRhf.error().message.find("stores a UHF state"), std::string::npos)
        << "message: " << uhfIntoRhf.error().message;
    std::filesystem::remove(uhfStore);
#endif
}

TEST(DriverErrorTest, TheDeviceRequirementReachesTheRecordAndUnhonourableOnesRefuse) {
    // The device axis, end to end through the pipeline a user's TOML takes. Two
    // facts, and the second is why the key is not merely a record change: a
    // requirement the run could not honour must be REFUSED, never executed
    // elsewhere - a run that computed on the host while its document said
    // "cuda:0" is exactly the substitution the key exists to forbid.
    //
    // The record half: `[builder] device = "host"` on an ordinary RHF run is
    // honoured by construction (the host is where this build's kernels run) and
    // the requirement appears in the document, in the selector vocabulary the
    // file wrote, so a consumer reads back a word it can write into a file.
    const auto hostRun = RunInputText(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.96], ["H", 0.9, 0.0, -0.3]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
[builder]
device = "host"
)");
    ASSERT_TRUE(hostRun.has_value()) << hostRun.error().message;
    // The KEY is asserted, not the bare word: "device" is also a legal VALUE in
    // this document (a compute-profile source), so a search for the bare word
    // would pass for a reason that has nothing to do with this axis.
    EXPECT_NE(hostRun->find("\"device\": \"host\""), std::string::npos) << *hostRun;

    // And an absent key states nothing: the member is ABSENT rather than
    // defaulted to "host", because a defaulted word would read as a requirement
    // the file never wrote.
    const auto noDevice = RunInputText(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.96], ["H", 0.9, 0.0, -0.3]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_TRUE(noDevice.has_value()) << noDevice.error().message;
    EXPECT_EQ(noDevice->find("\"device\":"), std::string::npos) << *noDevice;

    // The refusal half, and it takes TWO legs because the requirement is
    // answered at two sites that see different facts.
    //
    // (1) The parser's: a `cuda` requirement written with no backend word at
    // all. The block names no axis, so the selection resolves to its own
    // defaults - cpu among them - and the pair is contradictory in the file,
    // before any builder exists.
    const auto cudaWithoutBackend = RunInputText(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.96], ["H", 0.9, 0.0, -0.3]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
[builder]
device = "cuda:0"
)");
    ASSERT_FALSE(cudaWithoutBackend.has_value()) << *cudaWithoutBackend;
    EXPECT_EQ(cudaWithoutBackend.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(cudaWithoutBackend.error().message.find("builder.device = \"cuda:0\""),
              std::string::npos)
        << cudaWithoutBackend.error().message;

    // (2) The driver's, which the parser CANNOT reach: the two keys AGREE - the
    // backend axis says the device class and the device is a CUDA index - and the
    // requirement is still not honourable, because this build has no CUDA device
    // and the selection DEMOTES the `gpu` request to the ladder with its own
    // warning. So the run resolves to a host builder, and the requirement is
    // answered against what the run actually wired rather than against what the
    // file asked for. This is the leg that makes the driver's check a second
    // site rather than a copy of the parser's.
    const auto cudaOnDemotedGpu = RunInputText(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.96], ["H", 0.9, 0.0, -0.3]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
[builder]
execution_backend = "gpu"
device = "cuda:0"
)");
    ASSERT_FALSE(cudaOnDemotedGpu.has_value())
        << "a device requirement the run could not honour was executed elsewhere: "
        << *cudaOnDemotedGpu;
    EXPECT_EQ(cudaOnDemotedGpu.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(cudaOnDemotedGpu.error().message.find("builder.device = \"cuda:0\""),
              std::string::npos)
        << cudaOnDemotedGpu.error().message;
    EXPECT_NE(cudaOnDemotedGpu.error().message.find("the requirement cannot be honoured"),
              std::string::npos)
        << cudaOnDemotedGpu.error().message;
}

TEST(DriverErrorTest, ARestartPathBesideANonRestartGuessIsRefusedByName) {
    // The `guess.type` resolution point, pinned as a REFUSAL rather than a
    // silence. `guess.type` and `guess.restart_path` are one request between
    // them, and the path belongs to `restart` alone: the other three guess
    // words take no checkpoint file, so a path written beside one of them was
    // stored by `io` (which keeps what the file says, by its own contract)
    // and read by nothing at all - a key a user can write that the code
    // neither honoured, refused by name, nor demoted with a disclosure, which
    // is the knob-conformance defect's own definition.
    //
    // The pair is answered in ONE place (ResolveGuess, called from the
    // combination check), and this row reaches it through the pipeline the
    // way a user's TOML does. Both non-restart spellings are exercised,
    // because the rule is about the PAIR and not about one word of it: a rule
    // that only looked at the default would pass the first leg and fail the
    // second.
    const auto gwhStrayPath = RunInputText(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.96], ["H", 0.9, 0.0, -0.3]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
[guess]
type = "gwh"
restart_path = "stray.h5"
)");
    ASSERT_FALSE(gwhStrayPath.has_value())
        << "the stray path was silently ignored rather than answered: " << *gwhStrayPath;
    EXPECT_EQ(gwhStrayPath.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(gwhStrayPath.error().message.find("guess.restart_path"), std::string::npos)
        << gwhStrayPath.error().message;
    EXPECT_NE(gwhStrayPath.error().message.find("guess.type = \"restart\""), std::string::npos)
        << gwhStrayPath.error().message;

    const auto sadStrayPath = RunInputText(R"(
[molecule]
charge = 1
multiplicity = 2
atoms = [["O", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.96]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"
[guess]
type = "sad"
restart_path = "stray.h5"
)");
    ASSERT_FALSE(sadStrayPath.has_value()) << *sadStrayPath;
    EXPECT_EQ(sadStrayPath.error().code, qcx::ErrorCode::kInvalidArgument);

    // And the pair that IS the request still resolves: a `restart` run with a
    // path reaches the checkpoint read rather than this refusal. The read
    // itself fails on the missing file, which is a different error code and a
    // different message - so this leg separates "the pair is answered" from
    // "every path is refused". It needs the storage module: on a build
    // configured with QCX_ENABLE_IO=OFF the driver refuses the read with the
    // build's own cause before any store is touched, which names neither key.
#if defined(QcxHasStorage)
    const auto restartWithPath = RunInputText(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.96], ["H", 0.9, 0.0, -0.3]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
[guess]
type = "restart"
restart_path = "qcx_no_such_checkpoint_for_this_row.h5"
)");
    ASSERT_FALSE(restartWithPath.has_value());
    EXPECT_NE(restartWithPath.error().message.find("guess.restart_path"), std::string::npos)
        << restartWithPath.error().message;
    EXPECT_EQ(restartWithPath.error().message.find("was written beside a guess.type"),
              std::string::npos)
        << "the restart pair was answered by the pairing rule instead of by the read: "
        << restartWithPath.error().message;
#endif
}
