// TOML parsing for the qcx run schema. See parse_input.hpp
// for the strict-vocabulary/lenient-extra-keys policy.

#include "qcx/io/parse_input.hpp"

#include "qcx/grid/angular_grid.hpp"
#include "qcx/molecule/molecule.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <toml++/toml.hpp>

namespace qcx::io {
namespace {

// Builds a qcx::Error from a code and message.
qcx::Error Err(qcx::ErrorCode code, std::string message) {
    return qcx::Error{code, std::move(message)};
}

// Reads the whole file into a string; kInvalidArgument when unreadable.
qcx::Result<std::string> ReadFileText(const std::filesystem::path& path) {
    std::ifstream stream(path);

    if (!stream.is_open())
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument, "cannot open input file " + path.string()));
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

// One [symbol, x, y, z] atom row; kInvalidArgument for a wrong row shape,
// an empty symbol, or non-numeric coordinates.
//
// THE conversion site: this is the only place a file's coordinate unit is
// turned into the Bohr convention Molecule::Create wants, for any run through
// this parser. `unit` is the resolved [molecule] units value; kAngstrom
// multiplies each coordinate by qcx::molecule::kAngstromToBohr (the tree's
// one home for the factor), kBohr passes the parsed number through untouched,
// so a bohr file's geometry is bit-identical to the literals it wrote. A
// consumer that applies a factor of its own would be the second conversion
// this design exists to forbid (the 2026-08-26 pyscf-default and 2026-09-15
// Bohr-in-an-Angstrom-file hunts were the same mistake).
qcx::Result<RunAtom> ParseAtom(const toml::node_view<const toml::node>& row, CoordinateUnit unit) {
    const auto* array = row.as_array();

    if (array == nullptr || array->size() != 4)
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "each atom must be [symbol, x, y, z] (4 entries)"));
    }

    const auto symbol = (*array)[0].value<std::string>();

    if (!symbol.has_value() || symbol->empty())
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument, "atom symbol must be a non-empty string"));
    }

    const auto x = (*array)[1].value<double>();
    const auto y = (*array)[2].value<double>();
    const auto z = (*array)[3].value<double>();

    if (!x.has_value() || !y.has_value() || !z.has_value())
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument, "atom coordinates must be numbers"));
    }

    // A bohr row passes through unchanged; an angstrom row is converted. The
    // branch (rather than a always-applied factor of 1.0) makes the
    // pass-through literal: a bohr file's geometry is bit-identical to the
    // literals it wrote by construction, not by an argument about x * 1.0.
    const bool inAngstrom = unit == CoordinateUnit::kAngstrom;
    const double xBohr = inAngstrom ? *x * qcx::molecule::kAngstromToBohr : *x;
    const double yBohr = inAngstrom ? *y * qcx::molecule::kAngstromToBohr : *y;
    const double zBohr = inAngstrom ? *z * qcx::molecule::kAngstromToBohr : *z;

    return RunAtom{*symbol, xBohr, yBohr, zBohr};
}

// Maps one [molecule] units word to the io enum; kInvalidArgument with the
// accepted list on an unknown name. Both words are honoured - the key names a
// unit, and the parser can always apply either - so there is no third outcome
// here: an unknown or empty word is refused BY NAME rather than silently
// falling back to Angstrom, which is the failure mode the key exists to end
// (a file that says `units = "bohrs"` must not run as Angstrom).
qcx::Result<CoordinateUnit> ParseCoordinateUnit(std::string_view name) {
    if (name == "angstrom")
    {
        return CoordinateUnit::kAngstrom;
    }

    if (name == "bohr")
    {
        return CoordinateUnit::kBohr;
    }

    return std::unexpected(
        Err(qcx::ErrorCode::kInvalidArgument,
            "unknown molecule.units \"" + std::string(name) + "\" (angstrom | bohr)"));
}

// Maps one method.type word to the io enum; kInvalidArgument with the
// accepted list on an unknown name.
qcx::Result<MethodType> ParseMethod(std::string_view name) {
    if (name == "rhf")
    {
        return MethodType::kRhf;
    }

    if (name == "uhf")
    {
        return MethodType::kUhf;
    }

    if (name == "rks")
    {
        return MethodType::kRks;
    }

    if (name == "uks")
    {
        return MethodType::kUks;
    }

    return std::unexpected(
        Err(qcx::ErrorCode::kInvalidArgument,
            "unknown method.type \"" + std::string(name) + "\" (rhf | uhf | rks | uks)"));
}

// Maps one method.fock_builder word to the io enum; kInvalidArgument
// with the accepted list on an unknown name. The vocabulary has TWO axes
// (stated at BuilderKind, run_input.hpp): the FAMILY words are the
// enumerators below, and the TIER words are where the working set lives.
// The direct family's two tier words are handled differently only because
// they resolve differently:
//
//   - "lean" is the within-family lean member. It is no family word and no
//     enumerator (the kGpuSplit precedent), so the parse site intercepts it
//     into RunMethodInput::leanDirect before this table is consulted.
//   - "in_memory" names the tier kDirect ALREADY IS, so it is an alias and
//     this table maps it to kDirect - one enumerator, one behaviour, which is
//     why no record or refusal has to decide between the two spellings
//     (ToString's note: "direct" stays the canonical word).
//
// The tier words of the other families (`blocked`, `disk`) are not words of
// this key: they name rungs inside their own family's builder and are spelled
// at their own keys (RiTensorMode's "disk").
qcx::Result<BuilderKind> ParseBuilder(std::string_view name) {
    if (name == "direct")
    {
        return BuilderKind::kDirect;
    }

    // The tier axis's word for the same tier as "direct" (BuilderKind's
    // kDirect note). Accepted so a document can say WHERE the working set
    // lives instead of naming the family and leaving the tier inferred.
    if (name == "in_memory")
    {
        return BuilderKind::kDirect;
    }

    if (name == "ri_j_link")
    {
        return BuilderKind::kRiJLink;
    }

    if (name == "ri_jk")
    {
        return BuilderKind::kRiJk;
    }

    if (name == "qfmm")
    {
        return BuilderKind::kQfmm;
    }

    if (name == "gpu")
    {
        return BuilderKind::kGpu;
    }

    // The accepted list names the within-family "lean" word too: it is the
    // user's only in-band spelling help, and the parse site above this
    // table (ParseRunInput) is what intercepts it before this function is
    // consulted. `in_memory` is listed beside it because the two are the
    // tier axis's words, both accepted at this one key (this table maps
    // `in_memory` to kDirect; `lean` resolves to no enumerator).
    return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                               "unknown method.fock_builder \"" + std::string(name) +
                                   "\" (direct | ri_j_link | ri_jk | qfmm | gpu | lean | "
                                   "in_memory)"));
}

// Maps one method.ri_tensor_mode word to the io enum (the disk-rung
// knob); the rung
// selector has exactly two words, because a rung selector selects a rung.
// kInvalidArgument with the accepted list on an unknown name - and, for the
// retired force word, with the key that REPLACED it: a reader who wrote
// `forced_disk` is not making a typo, and an error that only lists the two
// surviving words would send them looking for a capability that moved rather
// than one that is gone (the force lives at
// [diagnostics] force_disk_ri; the record still speaks the word, see
// RunRiTensorMode::requested).
qcx::Result<RiTensorMode> ParseRiTensorMode(std::string_view name) {
    if (name == "auto")
    {
        return RiTensorMode::kAuto;
    }

    if (name == "disk")
    {
        return RiTensorMode::kDisk;
    }

    if (name == "forced_disk")
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                "method.ri_tensor_mode does not take \"forced_disk\": the forced disk mode is "
                "not a rung selection and is no longer spelled here. Set "
                "[diagnostics] force_disk_ri = true instead (the accepted rung words are "
                "auto | disk)"));
    }

    return std::unexpected(
        Err(qcx::ErrorCode::kInvalidArgument,
            "unknown method.ri_tensor_mode \"" + std::string(name) + "\" (auto | disk)"));
}

// Maps one method.accuracy word to the integrals preset; kInvalidArgument
// with the accepted list on an unknown name.
qcx::Result<qcx::integrals::AccuracyPreset> ParseAccuracy(std::string_view name) {
    if (name == "kLoose")
    {
        return qcx::integrals::AccuracyPreset::kLoose;
    }

    if (name == "kNormal")
    {
        return qcx::integrals::AccuracyPreset::kNormal;
    }

    if (name == "kTight")
    {
        return qcx::integrals::AccuracyPreset::kTight;
    }

    return std::unexpected(
        Err(qcx::ErrorCode::kInvalidArgument,
            "unknown method.accuracy \"" + std::string(name) + "\" (kLoose | kNormal | kTight)"));
}

// Maps one properties.esp word to the io enum; kInvalidArgument with the
// accepted list on an unknown name.
qcx::Result<EspFitScheme> ParseEspScheme(std::string_view name) {
    if (name == "chelpg")
    {
        return EspFitScheme::kChelpg;
    }

    if (name == "mk")
    {
        return EspFitScheme::kMerzKollman;
    }

    return std::unexpected(
        Err(qcx::ErrorCode::kInvalidArgument,
            "unknown properties.esp \"" + std::string(name) + "\" (chelpg | mk)"));
}

// Maps one guess.type word to the io enum; kInvalidArgument with the
// accepted list on an unknown name.
qcx::Result<GuessKind> ParseGuess(std::string_view name) {
    if (name == "core")
    {
        return GuessKind::kCore;
    }

    if (name == "gwh")
    {
        return GuessKind::kGwh;
    }

    if (name == "sad")
    {
        return GuessKind::kSad;
    }

    if (name == "restart")
    {
        return GuessKind::kRestart;
    }

    return std::unexpected(
        Err(qcx::ErrorCode::kInvalidArgument,
            "unknown guess.type \"" + std::string(name) + "\" (core | gwh | sad | restart)"));
}

// One schema scalar with an explicit type check: a present-but-wrong-typed
// key is kInvalidArgument (never a silent default), a missing key is
// nullopt and the caller applies the schema default.  The node_view's
// value<T>() conversion alone would silently swallow wrong types.
// The key argument is the qualified schema name ("molecule.charge"); the
// node_view subscript operates on the block's table, so the bare lookup
// key is everything after the last dot.
qcx::Result<std::optional<std::string>> ReadString(const toml::node_view<toml::node>& table,
                                                   std::string_view key) {
    const auto* node = table[key.substr(key.rfind('.') + 1)].node();

    if (node == nullptr)
    {
        return std::optional<std::string>{};
    }

    if (!node->is_string())
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument, "\"" + std::string(key) + "\" must be a string"));
    }

    // The is_string() check above makes the conversion safe: value<T>()
    // returns nullopt only on a type mismatch, which cannot happen here.
    return node->value<std::string>();
}

qcx::Result<std::optional<bool>> ReadBool(const toml::node_view<toml::node>& table,
                                          std::string_view key) {
    const auto* node = table[key.substr(key.rfind('.') + 1)].node();

    if (node == nullptr)
    {
        return std::optional<bool>{};
    }

    if (!node->is_boolean())
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument, "\"" + std::string(key) + "\" must be a bool"));
    }

    return node->value<bool>();
}

qcx::Result<std::optional<int>> ReadInt(const toml::node_view<toml::node>& table,
                                        std::string_view key) {
    const auto* node = table[key.substr(key.rfind('.') + 1)].node();

    if (node == nullptr)
    {
        return std::optional<int>{};
    }

    if (!node->is_integer())
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "\"" + std::string(key) + "\" must be an integer"));
    }

    return node->value<int>();
}

qcx::Result<std::optional<double>> ReadDouble(const toml::node_view<toml::node>& table,
                                              std::string_view key) {
    const auto* node = table[key.substr(key.rfind('.') + 1)].node();

    if (node == nullptr)
    {
        return std::optional<double>{};
    }

    // is_number() admits integer nodes too: TOML floats are the natural
    // spelling, but a bare integer (energy_tolerance = 1) is unambiguous.
    if (!node->is_number())
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument, "\"" + std::string(key) + "\" must be a number"));
    }

    return node->value<double>();
}

// ReadString with the required-key semantics: a missing or empty key is
// kInvalidArgument carrying the vocabulary hint, a wrong-typed key is
// reported by ReadString's type check.
qcx::Result<std::string> ReadRequiredString(const toml::node_view<toml::node>& table,
                                            // key is the schema path, vocabulary the accepted-value
                                            // hint - both string_views, deliberately not swappable.
                                            // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                            std::string_view key,
                                            std::string_view vocabulary) {
    auto value = ReadString(table, key);

    if (!value.has_value())
    {
        return std::unexpected(value.error());
    }

    if (!value->has_value() || (*value)->empty())
    {
        std::string message = "\"" + std::string(key) + "\" is required";

        if (!vocabulary.empty())
        {
            message += " (" + std::string(vocabulary) + ")";
        }

        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument, std::move(message)));
    }

    return **std::move(value);
}

// The shipped Lebedev sizes, " | "-separated, for the grid.angular_points
// refusal. Read from the module that owns them (grid/include/qcx/grid/
// angular_grid.hpp) rather than copied: the quadrature the engine builds IS
// that list (the two tables are one generator's output, and excgrid's mirror
// is what the XC grid build calls), so the parser's refusal and the engine's
// admission cannot disagree about which sizes exist. The message names them,
// so a reader is never left guessing which quadratures this build carries.
std::string LebedevSizeList() {
    std::string sizes;

    for (const std::size_t size : qcx::grid::AngularGrid::kAvailableSizes)
    {
        if (!sizes.empty())
        {
            sizes += " | ";
        }

        sizes += std::to_string(size);
    }

    return sizes;
}

} // namespace

// The inverse of ParseCoordinateUnit; see run_input.hpp ToString(CoordinateUnit).
// The words are the record's `molecule_units` vocabulary, so this function is
// what makes a run record name the unit its geometry was read under.
std::string_view ToString(CoordinateUnit unit) noexcept {
    switch (unit)
    {
    case CoordinateUnit::kAngstrom:
        return "angstrom";
    case CoordinateUnit::kBohr:
        return "bohr";
    }

    return "unknown";
}

// The inverse of ParseMethod; see run_input.hpp ToString(MethodType).
// The words are the ones the [method] type key accepts, so the parse's
// vocabulary and the record's `method` key (schema 34) are one vocabulary:
// a consumer reads the same four words on the input side and the output side,
// and the record's word is spelled by the parser's own home for it.
std::string_view ToString(MethodType method) noexcept {
    switch (method)
    {
    case MethodType::kRhf:
        return "rhf";
    case MethodType::kUhf:
        return "uhf";
    case MethodType::kRks:
        return "rks";
    case MethodType::kUks:
        return "uks";
    }

    return "unknown";
}

// The inverse of ParseBuilder; see run_input.hpp ToString(BuilderKind).
std::string_view ToString(BuilderKind builder) noexcept {
    switch (builder)
    {
    case BuilderKind::kDirect:
        return "direct";
    case BuilderKind::kRiJLink:
        return "ri_j_link";
    case BuilderKind::kRiJk:
        return "ri_jk";
    case BuilderKind::kQfmm:
        return "qfmm";
    case BuilderKind::kGpu:
        return "gpu";
    case BuilderKind::kGpuSplit:
        return "gpu_split";
    }

    return "unknown";
}

// The inverse of ParseAccuracy; see run_input.hpp ToString(AccuracyPreset).
// The words are the ones the input key accepts, so the parse's word and the
// record's word are one vocabulary.
std::string_view ToString(qcx::integrals::AccuracyPreset preset) noexcept {
    switch (preset)
    {
    case qcx::integrals::AccuracyPreset::kLoose:
        return "kLoose";
    case qcx::integrals::AccuracyPreset::kNormal:
        return "kNormal";
    case qcx::integrals::AccuracyPreset::kTight:
        return "kTight";
    }

    return "unknown";
}

// See run_input.hpp BuilderConsumesAux. The exhaustive-switch guard mirrors
// ResolveScfPath's (run_driver.cpp): with no `default:` arm and the
// unhandled-enumerator diagnostics promoted to errors, a kind this list does
// not place stops the build rather than inheriting the "no aux" answer - which
// is exactly what the equality it replaced gave every kind nobody had thought
// about yet, and what the checkpoint binding must not inherit (an aux the run
// used, missing from the stored fingerprint, is a restart that silently
// accepts a different configuration).
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(error : 4061)
#pragma warning(error : 4062)
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch"
#pragma clang diagnostic error "-Wswitch-enum"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif

bool BuilderConsumesAux(BuilderKind builder) noexcept {
    switch (builder)
    {
    case BuilderKind::kRiJLink:
    case BuilderKind::kRiJk:
        return true;

    case BuilderKind::kDirect:
    case BuilderKind::kQfmm:
    case BuilderKind::kGpu:
    case BuilderKind::kGpuSplit:
        return false;
    }

    // Reached only by a value no enumerator names (a programmatic call with an
    // out-of-range cast): the pre-fix derivation answered the same way
    // (`kind != kRiJLink`), so every such value keeps its exact answer.
    //
    // kRiJk is the ONE enumerator whose answer MOVED, when
    // the composed full-RI builder was wired and ri_jk became an aux
    // consumer in fact. The compatibility argument above - a checkpoint
    // written before the change must still load - does not reach it: the
    // driver refused every ri_jk request by name until that same increment, so
    // no run could write a checkpoint carrying this kind, and there is no
    // pre-change value for a load to disagree with. A future enumerator still
    // cannot move silently (the guarded switch above is a build failure), and
    // tests/parse_input_test.cpp states the exception explicitly rather than
    // looping over "every kind derives what it derived before".
    return false;
}

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__clang__)
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

// The inverse of ParseRiTensorMode; see run_input.hpp ToString(RiTensorMode).
std::string_view ToString(RiTensorMode mode) noexcept {
    switch (mode)
    {
    case RiTensorMode::kAuto:
        return "auto";
    case RiTensorMode::kDisk:
        return "disk";
    }

    return "unknown";
}

// ---------------------------------------------------------------------------
// THE ORTHOGONAL AXES. The axis table itself is
// stated once in run_input.hpp, above the three enums; what lives here is the
// request half of it - which words each axis accepts, and what each refusal is.

std::string_view ToString(IntegralFamily family) noexcept {
    switch (family)
    {
    case IntegralFamily::kDirect:
        return "direct";

    case IntegralFamily::kRiJLink:
        return "ri_j_link";

    case IntegralFamily::kRiJk:
        return "ri_jk";

    case IntegralFamily::kQfmm:
        return "qfmm";
    }

    return "direct";
}

std::string_view ToString(StorageTier tier) noexcept {
    switch (tier)
    {
    case StorageTier::kLean:
        return "lean";

    case StorageTier::kInMemory:
        return "in_memory";

    case StorageTier::kBlocked:
        return "blocked";

    case StorageTier::kDisk:
        return "disk";
    }

    return "in_memory";
}

std::string_view ToString(ExecutionBackend backend) noexcept {
    switch (backend)
    {
    case ExecutionBackend::kCpu:
        return "cpu";

    case ExecutionBackend::kGpu:
        return "gpu";

    case ExecutionBackend::kGpuSplit:
        return "gpu_split";
    }

    return "cpu";
}

bool operator==(const BuilderAxes& left, const BuilderAxes& right) noexcept {
    return left.integralFamily == right.integralFamily && left.storageTier == right.storageTier &&
           left.executionBackend == right.executionBackend;
}

BuilderAxes AxesOfBuilder(BuilderKind kind, bool leanMember) noexcept {
    BuilderAxes axes;

    switch (kind)
    {
    case BuilderKind::kDirect:
        // The one kind whose TIER is not decided by the kind: the direct family
        // has two members, and leanMember is the one statement of which runs
        // (ResolvedBuilderSelection::leanMember).
        axes.integralFamily = IntegralFamily::kDirect;
        axes.storageTier = leanMember ? StorageTier::kLean : StorageTier::kInMemory;
        axes.executionBackend = ExecutionBackend::kCpu;
        break;

    case BuilderKind::kRiJLink:
        axes.integralFamily = IntegralFamily::kRiJLink;
        axes.storageTier = StorageTier::kInMemory;
        axes.executionBackend = ExecutionBackend::kCpu;
        break;

    case BuilderKind::kRiJk:
        axes.integralFamily = IntegralFamily::kRiJk;
        axes.storageTier = StorageTier::kInMemory;
        axes.executionBackend = ExecutionBackend::kCpu;
        break;

    case BuilderKind::kQfmm:
        axes.integralFamily = IntegralFamily::kQfmm;
        axes.storageTier = StorageTier::kInMemory;
        axes.executionBackend = ExecutionBackend::kCpu;
        break;

    case BuilderKind::kGpu:
        // A BACKEND, not a family: GpuJkFockBuilder does the same preparations
        // as DirectJkFockBuilder::Create plus the device side, with the CPU
        // builder's screening semantics and its two-pass fp64 + certified-fp32
        // structure - the direct family's algorithm on a device.
        axes.integralFamily = IntegralFamily::kDirect;
        axes.storageTier = StorageTier::kInMemory;
        axes.executionBackend = ExecutionBackend::kGpu;
        break;

    case BuilderKind::kGpuSplit:
        // Named, never run: no builder exists, and the parser refuses the word.
        axes.integralFamily = IntegralFamily::kDirect;
        axes.storageTier = StorageTier::kInMemory;
        axes.executionBackend = ExecutionBackend::kGpuSplit;
        break;
    }

    return axes;
}

// The accepted words are the axis table's request column. Every word the table
// names that a request cannot reach is REFUSED here by name, with the reason and
// - where one exists - the key that owns it: the remedy-carrying refusal
// ParseRiTensorMode already gives the retired "forced_disk" word.
qcx::Result<IntegralFamily> ParseIntegralFamily(std::string_view name) {
    if (name == "direct")
    {
        return IntegralFamily::kDirect;
    }

    if (name == "ri_j_link")
    {
        return IntegralFamily::kRiJLink;
    }

    if (name == "ri_jk")
    {
        return IntegralFamily::kRiJk;
    }

    if (name == "qfmm")
    {
        return IntegralFamily::kQfmm;
    }

    return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                               "unknown builder.integral_family \"" + std::string(name) +
                                   "\" (direct | ri_j_link | ri_jk | qfmm)"));
}

// "lean" and "in_memory" are the two tiers the BUILDER SELECTION owns, which is
// why they are this key's and the axis is not: they are the direct family's
// within-family choice, and the deprecated key carried them beside its family
// words. The axis's other two values are refused, each by name and each with the
// reason it is not this key's.
qcx::Result<StorageTier> ParseStorageTier(std::string_view name) {
    if (name == "lean")
    {
        return StorageTier::kLean;
    }

    if (name == "in_memory")
    {
        return StorageTier::kInMemory;
    }

    if (name == "disk")
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                "builder.storage_tier = \"disk\" is not this key's: the disk tier is the last rung "
                "of the RI-J ladder, and the rung is selected by [method] ri_tensor_mode = "
                "\"disk\" (forced by [diagnostics] force_disk_ri). This key names the tier a "
                "builder SELECTION resolves - lean | in_memory - and a rung is a later decision "
                "by the engine's own estimate"));
    }

    if (name == "blocked")
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                "builder.storage_tier = \"blocked\" names a rung with no request surface: "
                "\"blocked\" is the full-RI family's accumulation rung "
                "(RiFullFockRung::kBlocked), named on the tier axis so the record can report it, "
                "and it cannot be asked for at any key (builder.storage_tier accepts lean | "
                "in_memory)"));
    }

    return std::unexpected(
        Err(qcx::ErrorCode::kInvalidArgument,
            "unknown builder.storage_tier \"" + std::string(name) + "\" (lean | in_memory)"));
}

// "gpu_split" is the axis's vocabulary-only value, the kGpuSplit precedent kept
// whole: the candidate has no builder to execute, so a REQUEST for it is
// refused here by name while the axis keeps the candidate nameable - and the
// kind itself stays refused in the driver (ValidateCombination) for a
// programmatic caller, which never passes through this table.
qcx::Result<ExecutionBackend> ParseExecutionBackend(std::string_view name) {
    if (name == "cpu")
    {
        return ExecutionBackend::kCpu;
    }

    if (name == "gpu")
    {
        return ExecutionBackend::kGpu;
    }

    if (name == "gpu_split")
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                "builder.execution_backend = \"gpu_split\" has no run path in v1: the word names "
                "the intra-build CPU+GPU batch-partition candidate, which has no builder to "
                "execute, and execution stays on the wired backends (cpu | gpu)"));
    }

    return std::unexpected(
        Err(qcx::ErrorCode::kInvalidArgument,
            "unknown builder.execution_backend \"" + std::string(name) + "\" (cpu | gpu)"));
}

// Resolves the [builder] block into the selection slots the wiring already
// reads - method.builder and the within-family leanDirect flag - so the axis
// vocabulary is a second SPELLING of one request, never a second state a run can
// be in (the `in_memory` alias rule, generalized to the whole selection).
// Nothing downstream changes: every consumer, every refusal and every record
// still reads one value, and the resolution is what keeps the two spellings
// from disagreeing.
//
// ONE REQUEST, ONE KEY. The axes and the deprecated [method] fock_builder
// key must not both be written, and the pair is REFUSED rather than checked for
// agreement: two keys naming one mechanism is the conflation the key split
// exists to keep apart, and an agreement check would have to elect one spelling
// as authoritative while silently leaving the other unread.
qcx::Result<void> ResolveBuilderBlock(RunInput& input, toml::table& table) {
    const auto builder = table["builder"];

    if (!builder.is_table())
    {
        return {};
    }

    auto familyName = ReadString(builder, "builder.integral_family");

    if (!familyName.has_value())
    {
        return std::unexpected(familyName.error());
    }

    auto tierName = ReadString(builder, "builder.storage_tier");

    if (!tierName.has_value())
    {
        return std::unexpected(tierName.error());
    }

    auto backendName = ReadString(builder, "builder.execution_backend");

    if (!backendName.has_value())
    {
        return std::unexpected(backendName.error());
    }

    const bool axesGiven =
        familyName->has_value() || tierName->has_value() || backendName->has_value();

    if (!axesGiven)
    {
        return {};
    }

    // The deprecated key's own spelling is what makes this a conflict, not the
    // slots: a programmatic caller can fill method.builder directly, and that is
    // not a second key in the file.
    if (input.builder.legacyFockBuilderWord.has_value())
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                "[builder] and [method] fock_builder both name the builder selection (the "
                "deprecated key wrote \"" +
                    *input.builder.legacyFockBuilderWord +
                    "\"). One request, one key: write the [builder] axes, or the deprecated "
                    "[method] fock_builder key alone"));
    }

    IntegralFamily family = IntegralFamily::kDirect;

    if (familyName->has_value())
    {
        auto parsed = ParseIntegralFamily(**familyName);

        if (!parsed.has_value())
        {
            return std::unexpected(parsed.error());
        }

        family = *parsed;
    }

    StorageTier tier = StorageTier::kInMemory;

    if (tierName->has_value())
    {
        auto parsed = ParseStorageTier(**tierName);

        if (!parsed.has_value())
        {
            return std::unexpected(parsed.error());
        }

        tier = *parsed;
    }

    ExecutionBackend backend = ExecutionBackend::kCpu;

    if (backendName->has_value())
    {
        auto parsed = ParseExecutionBackend(**backendName);

        if (!parsed.has_value())
        {
            return std::unexpected(parsed.error());
        }

        backend = *parsed;
    }

    // The acceptance matrix. An un-runnable COMBINATION is refused by name with
    // both sides stated - the same rule (a request that names a mechanism on a
    // family that cannot honour it) read across axes instead of across keys.
    if (tier == StorageTier::kLean && family != IntegralFamily::kDirect)
    {
        return std::unexpected(Err(
            qcx::ErrorCode::kInvalidArgument,
            "builder.storage_tier = \"lean\" is the direct family's within-family member, so it "
            "cannot be combined with builder.integral_family = \"" +
                std::string(ToString(family)) +
                "\" (lean is the direct family's Schwarz-only "
                "member; no other family has one)"));
    }

    if (tier == StorageTier::kLean && backend != ExecutionBackend::kCpu)
    {
        return std::unexpected(Err(
            qcx::ErrorCode::kInvalidArgument,
            "builder.storage_tier = \"lean\" with builder.execution_backend = \"" +
                std::string(ToString(backend)) +
                "\" is not a wired member: the lean tier is the direct family's CPU member, and "
                "the device builder has no lean arm"));
    }

    if (backend == ExecutionBackend::kGpu && family != IntegralFamily::kDirect)
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                "builder.execution_backend = \"gpu\" runs the DIRECT family on the device "
                "(GpuJkFockBuilder), so it cannot be combined with builder.integral_family = \"" +
                    std::string(ToString(family)) + "\""));
    }

    input.builder.integralFamily = family;
    input.builder.storageTier = tier;
    input.builder.executionBackend = backend;

    if (backend == ExecutionBackend::kGpu)
    {
        input.method.builder = BuilderKind::kGpu;
        return {};
    }

    switch (family)
    {
    case IntegralFamily::kDirect:

        // The one family with two members: the tier this key owns picks between
        // them, so `lean` never fills the family slot and `in_memory` does -
        // which is exactly what the deprecated key's "lean" carve-out did.
        if (tier == StorageTier::kLean)
        {
            input.method.leanDirect = true;
        } else
        {
            input.method.builder = BuilderKind::kDirect;
        }

        break;

    case IntegralFamily::kRiJLink:
        input.method.builder = BuilderKind::kRiJLink;
        break;

    case IntegralFamily::kRiJk:
        input.method.builder = BuilderKind::kRiJk;
        break;

    case IntegralFamily::kQfmm:
        input.method.builder = BuilderKind::kQfmm;
        break;
    }

    return {};
}

// The table a node holds, or null when it holds anything else. The visitor is
// what the parser library offers for a typed read, so "a table" is asked for
// by type: every other node type simply does not reach the walk below.
const toml::table* TableAt(const toml::node& node) {
    const toml::table* table = nullptr;

    node.visit([&table](const toml::table& element) { table = &element; });

    return table;
}

// Walks the whole document for the RI disk force key written outside its own
// table, and returns the dotted path it was found at, or an empty string when
// it is where it belongs.
//
// The key is the schema's own, and it names a mechanism: a file that wrote it
// under [method] meant the route to be forced, and a run that read past it
// would compute unforced while the author believed otherwise. So it is not
// covered by the tolerance for keys the schema does not name - it is refused
// by name, with the path it was written at and the one that is read, which is
// the treatment the retired rung word gets.
//
// The walk is over the document rather than a list of the tables this parser
// knows, so a table added later is covered without its own check. The dotted
// (method.force_disk_ri = true) and inline (method = { force_disk_ri = true })
// spellings need no case of their own: the parser normalizes both into a
// table, which this walk then sees like any other.
std::string FindMisplacedForceDiskKey(const toml::table& table, const std::string& prefix) {
    for (const auto& [key, node] : table)
    {
        const std::string name = std::string(key.str());
        // Appended rather than built as one `prefix + "." + name` chain: the
        // chain allocates a temporary per link, and a `const` result could not
        // be moved out of on the return below.
        std::string path = prefix;

        if (!prefix.empty())
        {
            path += ".";
        }

        path += name;

        // The force key's home: everything inside it is read where it belongs.
        if (path == "diagnostics")
        {
            continue;
        }

        // The key itself, at any depth: a top-level `force_disk_ri = true`
        // beside no table at all is the same claim as the one under [method].
        if (name == "force_disk_ri")
        {
            return path;
        }

        const auto* sub = TableAt(node);

        if (sub == nullptr)
        {
            continue;
        }

        std::string nested = FindMisplacedForceDiskKey(*sub, path);

        if (!nested.empty())
        {
            return nested;
        }
    }

    return {};
}

qcx::Result<RunInput> ParseRunInput(std::string_view tomlText) {
    toml::table table;

    try
    { table = toml::parse(tomlText); } catch (const toml::parse_error& error)
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                std::string("TOML syntax error: ") + std::string(error.description())));
    }

    RunInput input;

    // Before any block is read: a force key written anywhere but its own table
    // is refused for the whole file rather than left to the block that happens
    // to be parsed first.
    const std::string misplacedForceKey = FindMisplacedForceDiskKey(table, "");

    if (!misplacedForceKey.empty())
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                misplacedForceKey +
                    " is not read: the RI disk force is spelled in the [diagnostics] table "
                    "alone, and a copy of the key elsewhere forces nothing. Set "
                    "[diagnostics] force_disk_ri = true instead"));
    }

    const auto molecule = table["molecule"];

    if (!molecule.is_table())
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument, "missing [molecule] table"));
    }

    auto charge = ReadInt(molecule, "molecule.charge");

    if (!charge.has_value())
    {
        return std::unexpected(charge.error());
    }

    input.molecule.charge = charge->value_or(0);

    auto multiplicity = ReadInt(molecule, "molecule.multiplicity");

    if (!multiplicity.has_value())
    {
        return std::unexpected(multiplicity.error());
    }

    input.molecule.multiplicity = multiplicity->value_or(1);

    if (input.molecule.multiplicity < 1)
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument, "multiplicity must be at least 1"));
    }

    // The coordinate unit (the explicit-units packet). Optional and
    // type-strict; ABSENT means Angstrom, which is what every input file
    // written before this key existed means, so no existing file changes
    // meaning. Present means exactly what it says, and it is resolved HERE
    // because this is the file's boundary: the rows parsed below carry Bohr
    // (ParseAtom), `coordinateUnit` records the interpretation, and the run
    // record discloses it. An unknown word is refused BY NAME - a `units =
    // "bohr"` typo must not run a stretched molecule as Angstrom, which is
    // the 2026-09-15 hunt this key exists to end.
    auto unitsName = ReadString(molecule, "molecule.units");

    if (!unitsName.has_value())
    {
        return std::unexpected(unitsName.error());
    }

    if (unitsName->has_value())
    {
        auto unit = ParseCoordinateUnit(**unitsName);

        if (!unit.has_value())
        {
            return std::unexpected(unit.error());
        }

        input.molecule.coordinateUnit = *unit;
    }

    const auto atoms = molecule["atoms"].as_array();

    if (atoms == nullptr || atoms->empty())
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument, "molecule.atoms must be a non-empty array"));
    }

    for (const auto& row : *atoms)
    {
        auto atom =
            ParseAtom(toml::node_view<const toml::node>(row), input.molecule.coordinateUnit);

        if (!atom.has_value())
        {
            return std::unexpected(atom.error());
        }

        input.molecule.atoms.push_back(std::move(*atom));
    }

    const auto basis = table["basis"];

    if (!basis.is_table())
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument, "missing [basis] table"));
    }

    auto orbital = ReadRequiredString(basis, "basis.orbital", "");

    if (!orbital.has_value())
    {
        return std::unexpected(orbital.error());
    }

    input.basis.orbital = std::move(*orbital);

    auto aux = ReadString(basis, "basis.aux");

    if (!aux.has_value())
    {
        return std::unexpected(aux.error());
    }

    input.basis.aux = std::move(*aux);

    const auto method = table["method"];

    if (!method.is_table())
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument, "missing [method] table"));
    }

    auto methodName = ReadRequiredString(method, "method.type", "rhf | uhf | rks | uks");

    if (!methodName.has_value())
    {
        return std::unexpected(methodName.error());
    }

    // The two method-scoped Kohn-Sham keys. Both are
    // OPTIONAL and both stay unresolved here: io records the string and the
    // number the file carries, the driver resolves `functional` against the
    // functional registry and applies kDefaultScreeningTolerance when the
    // tolerance is absent. WHICH method words may carry them is deliberately
    // not this function's question - the validator owns that rule
    // (validate_input.cpp's XcKeyPolicyFor), so the key-presence policy has
    // exactly one home. Non-finite values are rejected the way the QFMM knobs
    // are: a nan or infinite tolerance would silently select a screening
    // behaviour the file does not appear to ask for (the theta precedent).
    auto functionalName = ReadString(method, "method.functional");

    if (!functionalName.has_value())
    {
        return std::unexpected(functionalName.error());
    }

    input.method.functional = *functionalName;
    auto screeningTolerance = ReadDouble(method, "method.screening_tolerance");

    if (!screeningTolerance.has_value())
    {
        return std::unexpected(screeningTolerance.error());
    }

    if (screeningTolerance->has_value() && !std::isfinite(**screeningTolerance))
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "method.screening_tolerance must be finite (0.0 selects the "
                                   "dense path)"));
    }

    if (screeningTolerance->has_value() && **screeningTolerance < 0.0)
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                "method.screening_tolerance must be non-negative (0.0 selects the dense path)"));
    }

    input.method.screeningTolerance = *screeningTolerance;

    // The builder key is optional: absent = auto (the direct-family
    // default, selection_resolution.hpp), any present value is
    // type-strict and vocabulary-checked (an empty string is not a
    // builder - omit the key for auto, never a silent "direct"). The
    // "lean" spelling is the direct family's TIER word for its within-family
    // lean member, so it never lands in the builder slot (no family word was
    // given): it sets the leanDirect flag below and the slot stays absent. No
    // new BuilderKind exists for it (the kGpuSplit no-word precedent). The
    // tier axis's other direct-family word, "in_memory", names the tier the
    // kDirect enumerator already is, so it needs no interception: ParseBuilder
    // maps it to kDirect and the slot carries the family that tier belongs to.
    auto builderName = ReadString(method, "method.fock_builder");

    if (!builderName.has_value())
    {
        return std::unexpected(builderName.error());
    }

    if (builderName->has_value())
    {
        // The DEPRECATED key's own word, recorded so the run record can say
        // which vocabulary the input used (schema 35). It is the conflict test
        // ResolveBuilderBlock reads below, and it is set here rather than
        // inferred there because this is the only place that knows the file
        // wrote the key at all - a programmatic caller filling the slot
        // directly has no spelling to record.
        input.builder.legacyFockBuilderWord = **builderName;

        if (**builderName == "lean")
        {
            input.method.leanDirect = true;
        } else
        {
            auto builder = ParseBuilder(**builderName);

            if (!builder.has_value())
            {
                return std::unexpected(builder.error());
            }

            input.method.builder = *builder;
        }
    }

    auto accuracyName = ReadRequiredString(method, "method.accuracy", "kLoose | kNormal | kTight");

    if (!accuracyName.has_value())
    {
        return std::unexpected(accuracyName.error());
    }

    auto methodType = ParseMethod(*methodName);

    if (!methodType.has_value())
    {
        return std::unexpected(methodType.error());
    }

    auto accuracy = ParseAccuracy(*accuracyName);

    if (!accuracy.has_value())
    {
        return std::unexpected(accuracy.error());
    }

    input.method.method = *methodType;
    input.method.accuracy = *accuracy;

    // The budget enforcement switch: optional and type-strict, absent = off.
    // The key is a behaviour change - the comparison it enables can move a
    // build's quartets from the fp32 lane to fp64 - so it is opt-in, and no
    // default, preset or builder word turns it on. The io layer stores
    // what the file says; the driver's direct-family machinery
    // wiring is its only consumer.
    // wiring is its only consumer.
    auto enforceCertifiedBound = ReadBool(method, "method.enforce_certified_bound");

    if (!enforceCertifiedBound.has_value())
    {
        return std::unexpected(enforceCertifiedBound.error());
    }

    input.method.enforceCertifiedBound = enforceCertifiedBound->value_or(false);

    // The certified fp32 lane's FORCE-ON request: optional and type-strict,
    // absent = false = the probe decides the lane's default. The two keys
    // are NOT the same switch and the parser keeps them apart:
    // enforce_certified_bound asks for the BUDGET ENFORCEMENT over the
    // lane's routing, force_certified_lane asks for the LANE. Both are
    // refused by name at the driver's resolution point on a route that
    // cannot honour them; the io layer stores what the file says and its
    // own contract is the type alone.
    auto forceCertifiedLane = ReadBool(method, "method.force_certified_lane");

    if (!forceCertifiedLane.has_value())
    {
        return std::unexpected(forceCertifiedLane.error());
    }

    input.method.forceCertifiedLane = forceCertifiedLane->value_or(false);

    // The optional QFMM knobs: the
    // QfmmOptions fields that existed but were unreachable from TOML.
    // Strict-parsed with the engine's own validation contract so a bad
    // value is rejected HERE, by its schema name, instead of deferred to a
    // builder error that names no TOML key. Every key is optional and
    // type-strict when present, for every run; the driver's qfmm path is
    // their only consumer (the "io stores what the file says"
    // layering).
    auto theta = ReadDouble(method, "method.theta");

    if (!theta.has_value())
    {
        return std::unexpected(theta.error());
    }

    // theta is the one knob the parse tightens past the engine's own
    // not-NaN check (the resources.memory_cap_gib precedent): +inf marks
    // every pair well-separated (the multipole expansion used where it is
    // invalid) and nan/-inf mark none (the far field silently vanishes) -
    // a non-finite value never means what the file appears to say. The
    // degenerate gate stays reachable at any finite theta < 0.
    if (theta->has_value() && !std::isfinite(**theta))
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument, "method.theta must be a finite number"));
    }

    input.method.theta = *theta;

    auto lMult = ReadInt(method, "method.l_mult");

    if (!lMult.has_value())
    {
        return std::unexpected(lMult.error());
    }

    // l_mult mirrors QfmmOptions.lMult: -1 = the accuracy preset's order,
    // 0..8 = the explicit order. The 8 is the engine's kQfmmMaxLMult cap
    // public in the qfmm_fock_build.hpp QfmmOptions doc text; the
    // bound moves with the cap, and Create() re-checks it regardless.
    if (lMult->has_value() && (**lMult < -1 || **lMult > 8))
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument, "method.l_mult must be in -1..8"));
    }

    input.method.lMult = *lMult;

    auto maxLeafSize = ReadInt(method, "method.max_leaf_size");

    if (!maxLeafSize.has_value())
    {
        return std::unexpected(maxLeafSize.error());
    }

    if (maxLeafSize->has_value() && **maxLeafSize < 1)
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument, "method.max_leaf_size must be >= 1"));
    }

    if (maxLeafSize->has_value())
    {
        input.method.maxLeafSize = static_cast<std::size_t>(**maxLeafSize);
    }

    auto crossover = ReadInt(method, "method.crossover_basis_function_count");

    if (!crossover.has_value())
    {
        return std::unexpected(crossover.error());
    }

    if (crossover->has_value() && **crossover < 0)
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "method.crossover_basis_function_count must be >= 0"));
    }

    input.method.crossoverBasisFunctionCount = *crossover;

    // The disk-rung knob: optional; the mode words are the io enum's own
    // vocabulary ("auto" | "disk" - the rung selector selects a rung and
    // nothing else). The FORCE is not a word here any more: it lives at
    // [diagnostics] force_disk_ri, and ParseRiTensorMode refuses the retired
    // word by name with that key as its remedy. The record's
    // kFastPath/kLightPath/kDisk names are the
    // DECISION vocabulary and never appear in the file.
    // DECISION vocabulary and never appear in the file.
    auto riTensorMode = ReadString(method, "method.ri_tensor_mode");

    if (!riTensorMode.has_value())
    {
        return std::unexpected(riTensorMode.error());
    }

    if (riTensorMode->has_value())
    {
        auto mode = ParseRiTensorMode(**riTensorMode);

        if (!mode.has_value())
        {
            return std::unexpected(mode.error());
        }

        input.method.riTensorMode = *mode;
    }

    // The RI-J orbit-expansion opt-in: optional
    // and type-strict, absent = false = the plain walk. The io layer stores
    // what the file says; the driver's ri_j_link arm is the consumer,
    // and the record's resources_resolved.ri_orbit_expansion carries the
    // outcome back (the ri_tensor_mode discipline: a request key whose
    // consumption is one branch deep must be answerable in the document).
    auto riOrbitExpansion = ReadBool(method, "method.ri_orbit_expansion");

    if (!riOrbitExpansion.has_value())
    {
        return std::unexpected(riOrbitExpansion.error());
    }

    input.method.riOrbitExpansion = *riOrbitExpansion;

    // The disk-tier ERI store request (the engine-decorator seam's request
    // surface): optional and type-strict. A NON-EMPTY path IS the
    // request (the scf.checkpoint_file / properties.molden spelling - a path
    // cannot be mistyped into another valid value), so empty and absent are
    // one not-requested state and neither installs a decorator. The io layer
    // stores what the file says; the driver owns the decorator (it is the
    // one module that may link storage) and the record's
    // resources_resolved.eri_store pairs the request with what ran.
    auto eriCacheStore = ReadString(method, "method.eri_cache_store");

    if (!eriCacheStore.has_value())
    {
        return std::unexpected(eriCacheStore.error());
    }

    input.method.eriCacheStore = eriCacheStore->value_or("");

    // The disk rung's chunk-size override (the DiskRiFockOptions.chunkBytes
    // passthrough): mirrors max_leaf_size - the io layer validates the
    // >= 1 floor by name, the size_t conversion happens after the check.
    auto riChunkBytes = ReadInt(method, "method.ri_chunk_bytes");

    if (!riChunkBytes.has_value())
    {
        return std::unexpected(riChunkBytes.error());
    }

    if (riChunkBytes->has_value() && **riChunkBytes < 1)
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument, "method.ri_chunk_bytes must be >= 1"));
    }

    if (riChunkBytes->has_value())
    {
        input.method.riChunkBytes = static_cast<std::size_t>(**riChunkBytes);
    }

    const auto scf = table["scf"];

    if (scf.is_table())
    {
        auto maxIterations = ReadInt(scf, "scf.max_iterations");

        if (!maxIterations.has_value())
        {
            return std::unexpected(maxIterations.error());
        }

        input.scf.maxIterations = maxIterations->value_or(100);

        auto energyTolerance = ReadDouble(scf, "scf.energy_tolerance");

        if (!energyTolerance.has_value())
        {
            return std::unexpected(energyTolerance.error());
        }

        input.scf.energyTolerance = energyTolerance->value_or(1e-8);

        auto densityTolerance = ReadDouble(scf, "scf.density_tolerance");

        if (!densityTolerance.has_value())
        {
            return std::unexpected(densityTolerance.error());
        }

        input.scf.densityTolerance = densityTolerance->value_or(1e-6);

        auto useDiis = ReadBool(scf, "scf.use_diis");

        if (!useDiis.has_value())
        {
            return std::unexpected(useDiis.error());
        }

        input.scf.useDiis = useDiis->value_or(true);

        auto traceFile = ReadString(scf, "scf.trace_file");

        if (!traceFile.has_value())
        {
            return std::unexpected(traceFile.error());
        }

        input.scf.traceFile = traceFile->value_or("");

        auto densityDump = ReadString(scf, "scf.density_dump");

        if (!densityDump.has_value())
        {
            return std::unexpected(densityDump.error());
        }

        input.scf.densityDumpFile = densityDump->value_or("");

        // The last-iterate checkpoint write (run-restart, RHF
        // only): a non-empty checkpoint_file asks the driver to save the
        // SCF state at the budget exit, and checkpoint_converged widens
        // that to the converged exit too. Both type-strict; the driver
        // rejects the key on the UHF path (nothing could consume it).
        auto checkpointFile = ReadString(scf, "scf.checkpoint_file");

        if (!checkpointFile.has_value())
        {
            return std::unexpected(checkpointFile.error());
        }

        input.scf.checkpointFile = checkpointFile->value_or("");

        auto checkpointConverged = ReadBool(scf, "scf.checkpoint_converged");

        if (!checkpointConverged.has_value())
        {
            return std::unexpected(checkpointConverged.error());
        }

        input.scf.checkpointConverged = checkpointConverged->value_or(false);
    }

    if (input.scf.maxIterations < 1 || input.scf.energyTolerance <= 0.0 ||
        input.scf.densityTolerance <= 0.0)
    {
        return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                   "scf.max_iterations must be >= 1 and the tolerances > 0"));
    }

    // The [resources] block: optional, each key optional when
    // absent but type-strict when present. memory_cap_gib is the hard
    // process-memory ceiling (0 = no input cap, the documented escape
    // hatch); thread_cap the OpenMP team ceiling (0 = all detected
    // hardware threads). A non-finite cap (TOML 1.0 nan/inf) cannot bound
    // a process - the >= 0 range check alone would admit it, so the
    // finiteness check is explicit.
    const auto resources = table["resources"];

    if (resources.is_table())
    {
        auto memoryCapGiB = ReadDouble(resources, "resources.memory_cap_gib");

        if (!memoryCapGiB.has_value())
        {
            return std::unexpected(memoryCapGiB.error());
        }

        if (memoryCapGiB->has_value() && !(std::isfinite(**memoryCapGiB) && **memoryCapGiB >= 0.0))
        {
            return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                       "resources.memory_cap_gib must be a finite number >= 0"));
        }

        input.resources.memoryCapGiB = memoryCapGiB->value_or(16.0);

        auto threadCap = ReadInt(resources, "resources.thread_cap");

        if (!threadCap.has_value())
        {
            return std::unexpected(threadCap.error());
        }

        if (threadCap->has_value() && **threadCap < 0)
        {
            return std::unexpected(
                Err(qcx::ErrorCode::kInvalidArgument, "resources.thread_cap must be >= 0"));
        }

        input.resources.threadCap = threadCap->value_or(0);
    }

    // The [memory_instrument] block: the per-term allocation
    // attribution opt-in; optional, each key optional when absent but
    // type-strict when present. enabled = true opens the instrumented
    // window at run entry (the driver's call); snapshot_interval_ms bounds
    // the watchdog's row cadence (>= 1); trace_file selects the
    // write-through trace (empty = stats-only, nothing written).
    const auto memoryInstrument = table["memory_instrument"];

    if (memoryInstrument.is_table())
    {
        auto enabled = ReadBool(memoryInstrument, "memory_instrument.enabled");

        if (!enabled.has_value())
        {
            return std::unexpected(enabled.error());
        }

        input.memoryInstrument.enabled = enabled->value_or(false);

        auto snapshotIntervalMs =
            ReadInt(memoryInstrument, "memory_instrument.snapshot_interval_ms");

        if (!snapshotIntervalMs.has_value())
        {
            return std::unexpected(snapshotIntervalMs.error());
        }

        if (snapshotIntervalMs->has_value() && **snapshotIntervalMs < 1)
        {
            return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                       "memory_instrument.snapshot_interval_ms must be >= 1"));
        }

        input.memoryInstrument.snapshotIntervalMs = snapshotIntervalMs->value_or(250);

        auto traceFile = ReadString(memoryInstrument, "memory_instrument.trace_file");

        if (!traceFile.has_value())
        {
            return std::unexpected(traceFile.error());
        }

        input.memoryInstrument.traceFile = traceFile->value_or("");
    }

    // The [diagnostics] block: the force-on
    // keys. Each is a separate key whose name says FORCE rather than a value
    // smuggled into a selector word - a selector carrying "force" is one key
    // naming two mechanisms (the key split). Every field optional and
    // default off, so an absent block changes nothing; io stores what the
    // file says and leaves what a route can honour to the driver's
    // resolution point.
    const auto diagnostics = table["diagnostics"];

    if (diagnostics.is_table())
    {
        auto forceDiskRi = ReadBool(diagnostics, "diagnostics.force_disk_ri");

        if (!forceDiskRi.has_value())
        {
            return std::unexpected(forceDiskRi.error());
        }

        input.diagnostics.forceDiskRi = forceDiskRi->value_or(false);
    }

    // The [symmetry] block: the full-group labeling stage switch,
    // strict vocabulary — full_group must be a bool when
    // present; absent = true (the stage runs by default).
    const auto symmetry = table["symmetry"];

    if (symmetry.is_table())
    {
        auto fullGroup = ReadBool(symmetry, "symmetry.full_group");

        if (!fullGroup.has_value())
        {
            return std::unexpected(fullGroup.error());
        }

        input.symmetry.fullGroup = fullGroup->value_or(true);
    }

    // The [grid] block: the XC integration grid's build parameters.
    // Optional, and every key optional inside it - an absent key resolves to
    // the engine's own compile-time default (RunGridInput, mirroring
    // XcGridSettings field for field), so an absent block builds exactly the
    // grid every run built before the block existed. The defaults are the
    // current behaviour STATED, never moved.
    //
    // The resolution happens here, ONCE: the driver's carry-all into
    // XcGridSettings is a shape change and not a second resolution (the
    // [molecule] units rule), so the six numbers the engine is created with
    // and the six the record discloses as `xc_grid` (schema 34) are the same
    // ones - which is what stops a run record from naming a grid it did not
    // use, and what makes a written key visible in the same place it acted.
    //
    // Every refusal names the key and the value it refused, and nothing is
    // clamped or snapped to the nearest legal value: a silently substituted
    // quadrature is a run on a grid the file did not ask for, disclosed as
    // though it had been. The floors mirror the engine's own admission
    // (excgrid refuses a zero radial count and a zero exponent) so a bad
    // number fails HERE, by name, instead of surfacing as a bare
    // kInvalidArgument from inside the grid build with no key to blame; alpha
    // and trim_weight get explicit finiteness and sign checks because excgrid
    // accepts them unchecked and both have a silent nonsense in reach
    // (alpha = 0 collapses every radial point onto the nucleus; a non-finite
    // trim_weight drops EVERY point).
    const auto grid = table["grid"];

    if (grid.is_table())
    {
        // Filled only when a key is present, and assigned as a whole below:
        // the struct's own defaults are the values an absent key keeps, and
        // the assignment is what tells the validator (and the record) that the
        // block was WRITTEN.
        RunGridInput gridSettings;

        auto radialPoints = ReadInt(grid, "grid.radial_points");

        if (!radialPoints.has_value())
        {
            return std::unexpected(radialPoints.error());
        }

        if (radialPoints->has_value() && **radialPoints < 1)
        {
            return std::unexpected(
                Err(qcx::ErrorCode::kInvalidArgument, "grid.radial_points must be >= 1"));
        }

        if (radialPoints->has_value())
        {
            gridSettings.radialPoints = static_cast<std::size_t>(**radialPoints);
        }

        auto angularPoints = ReadInt(grid, "grid.angular_points");

        if (!angularPoints.has_value())
        {
            return std::unexpected(angularPoints.error());
        }

        if (angularPoints->has_value())
        {
            const auto size = static_cast<std::size_t>(**angularPoints);
            const auto& available = qcx::grid::AngularGrid::kAvailableSizes;

            if (std::find(available.begin(), available.end(), size) == available.end())
            {
                return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                           "unknown grid.angular_points \"" +
                                               std::to_string(**angularPoints) + "\" (" +
                                               LebedevSizeList() + ")"));
            }

            gridSettings.angularPoints = size;
        }

        auto alpha = ReadDouble(grid, "grid.alpha");

        if (!alpha.has_value())
        {
            return std::unexpected(alpha.error());
        }

        if (alpha->has_value() && !(std::isfinite(**alpha) && **alpha > 0.0))
        {
            return std::unexpected(
                Err(qcx::ErrorCode::kInvalidArgument, "grid.alpha must be a finite number > 0"));
        }

        if (alpha->has_value())
        {
            gridSettings.alpha = **alpha;
        }

        auto radialExponent = ReadInt(grid, "grid.radial_exponent");

        if (!radialExponent.has_value())
        {
            return std::unexpected(radialExponent.error());
        }

        if (radialExponent->has_value() && **radialExponent < 1)
        {
            return std::unexpected(
                Err(qcx::ErrorCode::kInvalidArgument, "grid.radial_exponent must be >= 1"));
        }

        if (radialExponent->has_value())
        {
            gridSettings.radialExponent = static_cast<std::size_t>(**radialExponent);
        }

        auto trimWeight = ReadDouble(grid, "grid.trim_weight");

        if (!trimWeight.has_value())
        {
            return std::unexpected(trimWeight.error());
        }

        if (trimWeight->has_value() && !(std::isfinite(**trimWeight) && **trimWeight >= 0.0))
        {
            return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                       "grid.trim_weight must be a finite number >= 0"));
        }

        if (trimWeight->has_value())
        {
            gridSettings.trimWeight = **trimWeight;
        }

        auto blockTarget = ReadInt(grid, "grid.block_target");

        if (!blockTarget.has_value())
        {
            return std::unexpected(blockTarget.error());
        }

        if (blockTarget->has_value() && **blockTarget < 1)
        {
            return std::unexpected(
                Err(qcx::ErrorCode::kInvalidArgument, "grid.block_target must be >= 1"));
        }

        if (blockTarget->has_value())
        {
            gridSettings.blockTarget = static_cast<std::size_t>(**blockTarget);
        }

        input.grid = gridSettings;
    }

    const auto guess = table["guess"];
    auto guessName = ReadString(guess, "guess.type");

    if (!guessName.has_value())
    {
        return std::unexpected(guessName.error());
    }

    auto guessKind = ParseGuess(guessName->has_value() ? **guessName : "core");

    if (!guessKind.has_value())
    {
        return std::unexpected(guessKind.error());
    }

    input.guess = *guessKind;

    // The kRestart checkpoint path (guess.type = "restart"): restart_path
    // is the checkpoint file the driver loads and seeds the RHF
    // solver with. Type-strict whenever the [guess] table exists; required
    // to be non-empty exactly when the type is restart (any other type
    // tolerates an absent or empty path - the io layer stores what the
    // file says, the driver ignores it on the paths that do not read).
    auto restartPath = ReadString(guess, "guess.restart_path");

    if (!restartPath.has_value())
    {
        return std::unexpected(restartPath.error());
    }

    input.guessRestartPath = restartPath->value_or("");

    if (*guessKind == GuessKind::kRestart && input.guessRestartPath.empty())
    {
        return std::unexpected(
            Err(qcx::ErrorCode::kInvalidArgument,
                "guess.type = \"restart\" needs a non-empty guess.restart_path"));
    }

    // The [properties] block: all analyses default off; the
    // block is optional, and within it every key is optional when absent
    // but type-strict when present.
    const auto properties = table["properties"];

    if (properties.is_table())
    {
        auto hirshfeld = ReadBool(properties, "properties.hirshfeld");

        if (!hirshfeld.has_value())
        {
            return std::unexpected(hirshfeld.error());
        }

        input.properties.hirshfeld = hirshfeld->value_or(false);

        auto voronoi = ReadBool(properties, "properties.voronoi");

        if (!voronoi.has_value())
        {
            return std::unexpected(voronoi.error());
        }

        input.properties.voronoi = voronoi->value_or(false);

        auto eddb = ReadBool(properties, "properties.eddb");

        if (!eddb.has_value())
        {
            return std::unexpected(eddb.error());
        }

        input.properties.eddb = eddb->value_or(false);

        auto fukui = ReadBool(properties, "properties.fukui");

        if (!fukui.has_value())
        {
            return std::unexpected(fukui.error());
        }

        input.properties.fukui = fukui->value_or(false);

        auto nalewajski = ReadBool(properties, "properties.nalewajski");

        if (!nalewajski.has_value())
        {
            return std::unexpected(nalewajski.error());
        }

        input.properties.nalewajski = nalewajski->value_or(false);

        auto densityAtNuclei = ReadBool(properties, "properties.density_at_nuclei");

        if (!densityAtNuclei.has_value())
        {
            return std::unexpected(densityAtNuclei.error());
        }

        input.properties.densityAtNuclei = densityAtNuclei->value_or(false);

        auto qtaim = ReadBool(properties, "properties.qtaim");

        if (!qtaim.has_value())
        {
            return std::unexpected(qtaim.error());
        }

        input.properties.qtaim = qtaim->value_or(false);

        auto moldenPath = ReadString(properties, "properties.molden");

        if (!moldenPath.has_value())
        {
            return std::unexpected(moldenPath.error());
        }

        // An empty value leaves the key unset - absent = off, the path-key
        // contract (unlike properties.esp, where the empty string is a
        // typo and rejected: a molden path cannot be mistyped into a
        // different valid value, so empty means "not requested").
        input.properties.molden = moldenPath->value_or("");

        auto espName = ReadString(properties, "properties.esp");

        if (!espName.has_value())
        {
            return std::unexpected(espName.error());
        }

        if (espName->has_value())
        {
            // An empty string is a typo for the enum, not a request to
            // skip the block: before the check the empty
            // value was silently treated as unset, indistinguishable from
            // the omitted key in the JSON output.
            if ((*espName)->empty())
            {
                return std::unexpected(
                    Err(qcx::ErrorCode::kInvalidArgument, "\"properties.esp\" must not be empty"));
            }

            auto scheme = ParseEspScheme(**espName);

            if (!scheme.has_value())
            {
                return std::unexpected(scheme.error());
            }

            input.properties.esp = *scheme;
        }

        const auto* fragments = properties["nocv_fragments"].as_array();

        if (fragments != nullptr)
        {
            for (const auto& row : *fragments)
            {
                const auto* group = row.as_array();

                if (group == nullptr || group->empty())
                {
                    return std::unexpected(Err(qcx::ErrorCode::kInvalidArgument,
                                               "each properties.nocv_fragments group must be "
                                               "a non-empty array of atom indices"));
                }

                std::vector<std::size_t> groupIndices;

                for (const auto& index : *group)
                {
                    const auto value = index.value<std::int64_t>();

                    if (!value.has_value() || *value < 0)
                    {
                        return std::unexpected(
                            Err(qcx::ErrorCode::kInvalidArgument,
                                "properties.nocv_fragments indices must be non-negative "
                                "integers"));
                    }

                    groupIndices.push_back(static_cast<std::size_t>(*value));
                }

                input.properties.nocvFragments.push_back(std::move(groupIndices));
            }
        }
    }

    // The orthogonal axis vocabulary, resolved
    // LAST so it reads the deprecated key's own slot: the two spellings of the
    // selection are one request, and this is where they are held against each
    // other and against the acceptance matrix. Placed at the end of the function
    // because every slot it writes is one nothing above it has read yet.
    auto builderResolved = ResolveBuilderBlock(input, table);

    if (!builderResolved.has_value())
    {
        return std::unexpected(builderResolved.error());
    }

    return input;
}

qcx::Result<RunInput> ParseRunInputFile(const std::filesystem::path& path) {
    auto text = ReadFileText(path);

    if (!text.has_value())
    {
        return std::unexpected(text.error());
    }

    return ParseRunInput(*text);
}

} // namespace qcx::io
