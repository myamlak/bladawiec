// The `qcx run --help` text is a CLAIM ABOUT THE PARSER, and this file is
// where the claim is checked (the text itself lives in
// driver/src/internal/cli_help_text.hpp, which the CLI prints verbatim).
//
// WHAT IS PINNED AND WHAT IS DERIVED. Nothing here holds a copy of the key
// list, of a vocabulary or of a default: the key set is read out of the parser
// source's own read sites, a closed vocabulary is discovered by probing the
// parser with a value no vocabulary holds and reading the words off its
// refusal, and a default is compared against the schema struct's own field
// (or, for a key the engine consumes, the engine's option struct). So a key
// added to the parser without a help row, a word added to a vocabulary, a
// default that moves in its struct - each fails here, and none of them needs
// this file to change. That is the whole point: a test that compared the help
// against a second copy of the key list would pin the copy and drift with it.

#include "internal/cli_help_text.hpp"
#include "qcx/integrals/qfmm_fock_build.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/io/run_input.hpp"
#include "qcx/io/validate_input.hpp"

#ifdef QcxHasStorage
#include <qcx/storage/disk_ri_fock_build.hpp>
#endif

#include <charconv>
#include <cstddef>
#include <fstream>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// The help text, read as rows (the format contract is stated at the top of
// cli_help_text.hpp, and the parser below is that contract written down).
// ---------------------------------------------------------------------------

struct HelpRow {
    std::string block; // the [block] the row sits under
    std::string key; // the bare key, as the file spells it
    std::string shown; // the value the help shows: a default, or an example
    std::string description; // the description, continuation lines joined in
};

struct HelpText {
    std::vector<std::string> blocks;
    std::vector<HelpRow> rows;
    std::vector<std::string> problems; // ways the text broke its own format
};

std::string Trimmed(std::string_view text) {
    const std::size_t first = text.find_first_not_of(" \t\r");

    if (first == std::string_view::npos)
    {
        return {};
    }

    const std::size_t last = text.find_last_not_of(" \t\r");

    return std::string(text.substr(first, last - first + 1));
}

std::vector<std::string> SplitLines(std::string_view text) {
    std::vector<std::string> lines;
    std::size_t start = 0;

    while (start <= text.size())
    {
        const std::size_t end = text.find('\n', start);
        const std::size_t stop = (end == std::string_view::npos) ? text.size() : end;
        lines.push_back(std::string(text.substr(start, stop - start)));

        if (end == std::string_view::npos)
        {
            break;
        }

        start = end + 1;
    }

    return lines;
}

std::string BlockKey(const HelpRow& row) {
    return row.block + "." + row.key;
}

// A shown string value as the bare word it names: the help shows a string the
// way the file spells it ("rhf"), while a vocabulary is the words themselves
// (rhf).
std::string Unquoted(const std::string& shown) {
    if (shown.size() >= 2 && shown.front() == '"' && shown.back() == '"')
    {
        return shown.substr(1, shown.size() - 2);
    }

    return shown;
}

// The indent a key row opens at. A line indented deeper continues the previous
// row's description; a line at this indent is a key row; anything shallower is
// the prose between the blocks.
constexpr std::size_t kKeyRowIndent = 4;

HelpText ParseHelp(std::string_view text) {
    HelpText help;
    std::string block;

    for (const std::string& line : SplitLines(text))
    {
        const std::string trimmed = Trimmed(line);

        if (trimmed.empty())
        {
            continue;
        }

        if (line[0] == '[')
        {
            const std::size_t close = line.find(']');

            if (close == std::string::npos)
            {
                help.problems.push_back("a block header has no closing bracket: " + line);
                continue;
            }

            block = line.substr(1, close - 1);
            help.blocks.push_back(block);
            continue;
        }

        const std::size_t indent = line.find_first_not_of(' ');

        if (indent > kKeyRowIndent)
        {
            if (help.rows.empty())
            {
                help.problems.push_back("an indented line follows no key row: " + line);
                continue;
            }

            help.rows.back().description += " " + trimmed;
            continue;
        }

        if (indent != kKeyRowIndent)
        {
            continue; // the usage line and the prose between blocks
        }

        const std::size_t equals = line.find(" = ", kKeyRowIndent);

        if (equals == std::string::npos)
        {
            help.problems.push_back("a line at the key indent is not a key row: " + line);
            continue;
        }

        const std::string key = line.substr(kKeyRowIndent, equals - kKeyRowIndent);

        if (key.empty() || key.find(' ') != std::string::npos)
        {
            help.problems.push_back("a key row has no bare key: " + line);
            continue;
        }

        if (block.empty())
        {
            help.problems.push_back("a key row sits under no block header: " + line);
            continue;
        }

        // The value ends at the first run of two spaces; what follows is the
        // description (which a continuation line may extend).
        const std::string rest = line.substr(equals + 3);
        const std::size_t gap = rest.find("  ");
        const std::string shown =
            (gap == std::string::npos) ? Trimmed(rest) : Trimmed(rest.substr(0, gap));
        const std::string description =
            (gap == std::string::npos) ? std::string{} : Trimmed(rest.substr(gap));
        help.rows.push_back(HelpRow{block, key, shown, description});
    }

    return help;
}

const HelpRow* FindRow(const HelpText& help, std::string_view key) {
    for (const HelpRow& row : help.rows)
    {
        if (BlockKey(row) == key)
        {
            return &row;
        }
    }

    return nullptr;
}

// ---------------------------------------------------------------------------
// The parser's key set, read out of the source that spells it.
// ---------------------------------------------------------------------------
//
// An unknown key is IGNORED by the parser (there is no unknown-key scan in it
// to fail), so no runtime probe can tell "the parser reads this key" from "the
// parser never looks". Existence is a property of the source, so that is where
// this reads it: io/src/parse_input.cpp is the one file where every key the
// schema accepts is named - the read helpers carry the dotted "block.key"
// spelling, and the two array-valued keys are read by subscript.

constexpr const char* kParserSource = "io/src/parse_input.cpp";

// The tree root, baked by driver/CMakeLists.txt (the same device the driver
// library uses for data/basis). A source-scanning test has to name its source.
constexpr const char* kSourceDir = QcxSourceDir;

struct ParserKeys {
    std::set<std::string> all;
    std::set<std::string> required; // refused when omitted
    bool readable = false; // false = the source could not be read
};

// `//` comments are stripped first, quote-aware, so a commented-out read site
// does not count as an accepted key and a literal's own text is never cut in
// half. The file holds no raw string literals and no block comments, so a
// quote-and-escape scan is exact for it.
std::string WithoutComments(std::string_view source) {
    std::string code;
    code.reserve(source.size());
    bool inString = false;
    bool escaped = false;

    for (std::size_t i = 0; i < source.size(); ++i)
    {
        const char c = source[i];

        if (inString)
        {
            code.push_back(c);

            if (escaped)
            {
                escaped = false;
            } else if (c == '\\')
            {
                escaped = true;
            } else if (c == '"')
            { inString = false; }

            continue;
        }

        if (c == '"')
        {
            inString = true;
            code.push_back(c);
            continue;
        }

        if (c == '/' && i + 1 < source.size() && source[i + 1] == '/')
        {
            while (i < source.size() && source[i] != '\n')
            {
                ++i;
            }

            if (i < source.size())
            {
                code.push_back('\n');
            }

            continue;
        }

        code.push_back(c);
    }

    return code;
}

ParserKeys ScanParserKeys() {
    ParserKeys keys;
    std::ifstream file(std::string(kSourceDir) + "/" + kParserSource, std::ios::binary);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    const std::string source = buffer.str();

    if (source.empty())
    {
        return keys;
    }

    keys.readable = true;
    const std::string code = WithoutComments(source);

    const std::regex dotted(R"rx("([a-z][a-z0-9_]*\.[a-z][a-z0-9_]*))rx");
    const std::regex subscript(R"rx(([a-z_]+)\["([a-z][a-z0-9_]*)"\]\.as_array\(\))rx");
    const std::regex required(
        R"rx(ReadRequiredString\([^,]+,\s*"([a-z][a-z0-9_]*\.[a-z][a-z0-9_]*))rx");

    for (std::sregex_iterator it(code.begin(), code.end(), dotted), end; it != end; ++it)
    {
        keys.all.insert((*it)[1].str());
    }

    for (std::sregex_iterator it(code.begin(), code.end(), subscript), end; it != end; ++it)
    {
        keys.all.insert((*it)[1].str() + "." + (*it)[2].str());
    }

    for (std::sregex_iterator it(code.begin(), code.end(), required), end; it != end; ++it)
    {
        keys.required.insert((*it)[1].str());
    }

    return keys;
}

// ---------------------------------------------------------------------------
// Documents built out of the help's own rows.
// ---------------------------------------------------------------------------
//
// Every check below asks the parser about one key at a time, holding every
// other key at the value the help shows for it. So the document is assembled
// from the help text itself: a check that fails is a disagreement between the
// help and the parser, never a fixture this file invented.

using Values = std::map<std::string, std::string>;

std::string BuildDocument(const Values& values) {
    std::string text;
    std::string openBlock;

    for (const auto& [key, value] : values)
    {
        const std::size_t dot = key.find('.');
        const std::string block = key.substr(0, dot);

        if (block != openBlock)
        {
            text += "[" + block + "]\n";
            openBlock = block;
        }

        text += key.substr(dot + 1) + " = " + value + "\n";
    }

    return text;
}

// The pieces every check starts from: the help parsed into rows, the parser's
// key set, and the document every probe is a variation of - one that writes
// every key the parser REFUSES to see omitted, at the value the help shows for
// it. The required keys come from the scan, so a key the parser starts to
// require (or stops requiring) moves the document with no line changing here.
struct Fixture {
    HelpText help;
    ParserKeys keys;
    Values base;
    std::vector<std::string> problems;
    bool usable = false;
};

Fixture BuildFixture() {
    Fixture fixture;
    fixture.help = ParseHelp(qcx::driver::internal::kRunHelpText);
    fixture.keys = ScanParserKeys();

    if (!fixture.keys.readable)
    {
        fixture.problems.push_back("cannot read " + std::string(kSourceDir) + "/" + kParserSource);
        return fixture;
    }

    if (fixture.keys.required.empty())
    {
        fixture.problems.push_back("the scan of " + std::string(kParserSource) +
                                   " found no required key");
        return fixture;
    }

    // The atom rows are required by SHAPE (the parser refuses an empty list)
    // rather than by a helper call, so this one key is taken by name.
    const HelpRow* atoms = FindRow(fixture.help, "molecule.atoms");

    if (atoms == nullptr)
    {
        fixture.problems.push_back("the help documents no molecule.atoms row");
    } else
    {
        fixture.base["molecule.atoms"] = atoms->shown;
    }

    for (const std::string& key : fixture.keys.required)
    {
        const HelpRow* row = FindRow(fixture.help, key);

        if (row == nullptr)
        {
            fixture.problems.push_back("the parser requires " + key +
                                       " and the help has no row for it");
            continue;
        }

        fixture.base[key] = row->shown;
    }

    fixture.usable = fixture.problems.empty();
    return fixture;
}

// Whatever BuildFixture could not turn into a failed check by itself.
void Report(const Fixture& fixture) {
    for (const std::string& problem : fixture.problems)
    {
        ADD_FAILURE() << problem;
    }
}

// ---------------------------------------------------------------------------
// Closed vocabularies, discovered rather than listed.
// ---------------------------------------------------------------------------

// The values a probe writes when it asks the parser "is this a closed
// vocabulary?": no vocabulary holds either, so the words in the refusal are
// the vocabulary and any other outcome is not one.
constexpr const char* kProbeValues[] = {"\"no-such-value\"", "999999999"};

// The parser's own refusal shape for an unknown word:
//   unknown <schema.path> "<value>" (<accepted words>)
// which is why the help spells its words with " | " - the parser's separator.
const char* const kRefusalPattern = R"rx(unknown ([A-Za-z0-9_.]+) "[^"]*" \(([^)]*)\)$)rx";

std::map<std::string, std::string> ProbeVocabularies(const ParserKeys& keys,
                                                     const Values& base,
                                                     std::vector<std::string>& problems) {
    static const std::regex refusal(kRefusalPattern);
    std::map<std::string, std::string> vocabularies;

    for (const std::string& key : keys.all)
    {
        for (const char* probe : kProbeValues)
        {
            Values values = base;
            values[key] = probe;

            const auto parsed = qcx::io::ParseRunInput(BuildDocument(values));

            if (parsed.has_value())
            {
                continue;
            }

            std::smatch match;

            if (!std::regex_match(parsed.error().message, match, refusal))
            {
                continue;
            }

            if (match[1].str() != key)
            {
                problems.push_back("a probe of " + key + " was refused about " + match[1].str() +
                                   ": " + parsed.error().message);
                continue;
            }

            vocabularies[key] = match[2].str();
        }
    }

    return vocabularies;
}

std::set<std::string> SplitWords(std::string_view words) {
    std::set<std::string> split;
    std::size_t start = 0;

    while (start <= words.size())
    {
        const std::size_t end = words.find(" | ", start);
        const std::size_t stop = (end == std::string_view::npos) ? words.size() : end;
        const std::string token = Trimmed(words.substr(start, stop - start));

        if (!token.empty())
        {
            split.insert(token);
        }

        if (end == std::string_view::npos)
        {
            break;
        }

        start = end + 3;
    }

    return split;
}

bool IsWordChar(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

// Whether a token of prose holds a word as a word: delimited by non-word
// characters on both sides, so "One of: direct" holds `direct`,
// "in_memory. Absent = the size ladder" still holds `in_memory`, and the
// `direct` inside a longer word holds nothing.
bool HoldsWord(std::string_view token, std::string_view word) {
    std::size_t at = token.find(word, 0);

    while (at != std::string_view::npos)
    {
        const bool leftFree = (at == 0) || !IsWordChar(token[at - 1]);
        const std::size_t after = at + word.size();
        const bool rightFree = (after >= token.size()) || !IsWordChar(token[after]);

        if (leftFree && rightFree)
        {
            return true;
        }

        at = token.find(word, at + 1);
    }

    return false;
}

std::vector<std::string> SplitOnPipe(std::string_view text) {
    std::vector<std::string> tokens;
    std::size_t start = 0;

    while (start <= text.size())
    {
        const std::size_t end = text.find(" | ", start);
        const std::size_t stop = (end == std::string_view::npos) ? text.size() : end;
        tokens.push_back(Trimmed(text.substr(start, stop - start)));

        if (end == std::string_view::npos)
        {
            break;
        }

        start = end + 3;
    }

    return tokens;
}

// The words a description presents as a LIST for one vocabulary: a run of
// consecutive " | "-separated tokens, one accepted word each, whose words are
// exactly the parser's own. The description's prose may name the words freely
// elsewhere - what is checked is the list the format contract asks for, which
// is what a reader copies from.
std::optional<std::set<std::string>> HelpWordsFor(std::string_view description,
                                                  const std::string& parserWords) {
    const std::set<std::string> wanted = SplitWords(parserWords);

    if (wanted.empty())
    {
        return std::nullopt;
    }

    const std::vector<std::string> tokens = SplitOnPipe(description);

    for (std::size_t start = 0; start + wanted.size() <= tokens.size(); ++start)
    {
        std::set<std::string> found;
        bool everyTokenHoldsAWord = true;

        for (std::size_t i = start; i < start + wanted.size(); ++i)
        {
            bool holds = false;

            for (const std::string& word : wanted)
            {
                if (HoldsWord(tokens[i], word))
                {
                    found.insert(word);
                    holds = true;
                }
            }

            if (!holds)
            {
                everyTokenHoldsAWord = false;
                break;
            }
        }

        if (everyTokenHoldsAWord && found == wanted)
        {
            return found;
        }
    }

    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Defaults, compared against the struct that owns them.
// ---------------------------------------------------------------------------

// The two documents a default claim compares, as one value with named fields
// rather than as two adjacent `const RunInput&` parameters: a claim compares
// the same member on both documents, so it reads the same whichever way round
// a caller passed them, and only the type can say which document is which.
struct DefaultPair {
    const qcx::io::RunInput& written; // the document that wrote the key
    const qcx::io::RunInput& omitted; // the document that omitted it
};

// One default the help claims: `matches` is true exactly when the document
// that WROTE the help's shown value leaves the schema (or the engine's option
// struct) where the document that OMITTED the key left it.
struct DefaultClaim {
    std::string_view key;
    bool (*matches)(const DefaultPair& documents);
};

const DefaultClaim kDefaultClaims[] = {
    {"molecule.charge",
     [](const DefaultPair& documents) {
         return documents.written.molecule.charge == documents.omitted.molecule.charge;
     }},
    {"molecule.multiplicity",
     [](const DefaultPair& documents) {
         return documents.written.molecule.multiplicity == documents.omitted.molecule.multiplicity;
     }},
    {"molecule.units",
     [](const DefaultPair& documents) {
         return documents.written.molecule.coordinateUnit ==
                documents.omitted.molecule.coordinateUnit;
     }},
    {"method.type",
     [](const DefaultPair& documents) {
         return documents.written.method.method == documents.omitted.method.method;
     }},
    {"method.accuracy",
     [](const DefaultPair& documents) {
         return documents.written.method.accuracy == documents.omitted.method.accuracy;
     }},
    {"method.enforce_certified_bound",
     [](const DefaultPair& documents) {
         return documents.written.method.enforceCertifiedBound ==
                documents.omitted.method.enforceCertifiedBound;
     }},
    {"method.force_certified_lane",
     [](const DefaultPair& documents) {
         return documents.written.method.forceCertifiedLane ==
                documents.omitted.method.forceCertifiedLane;
     }},
    // The passthrough keys whose engine default is itself writable are
    // compared against the engine's own option struct further down; the two
    // here are the optional-around-the-request kind, where an absent key and
    // the shown value are the same request.
    {"method.ri_tensor_mode",
     [](const DefaultPair& documents) {
         return documents.written.method.riTensorMode.value_or(qcx::io::RiTensorMode::kAuto) ==
                documents.omitted.method.riTensorMode.value_or(qcx::io::RiTensorMode::kAuto);
     }},
    {"method.ri_orbit_expansion",
     [](const DefaultPair& documents) {
         return documents.written.method.riOrbitExpansion.value_or(false) ==
                documents.omitted.method.riOrbitExpansion.value_or(false);
     }},
    {"method.eri_cache_store",
     [](const DefaultPair& documents) {
         return documents.written.method.eriCacheStore == documents.omitted.method.eriCacheStore;
     }},
    {"scf.max_iterations",
     [](const DefaultPair& documents) {
         return documents.written.scf.maxIterations == documents.omitted.scf.maxIterations;
     }},
    {"scf.energy_tolerance",
     [](const DefaultPair& documents) {
         return documents.written.scf.energyTolerance == documents.omitted.scf.energyTolerance;
     }},
    {"scf.density_tolerance",
     [](const DefaultPair& documents) {
         return documents.written.scf.densityTolerance == documents.omitted.scf.densityTolerance;
     }},
    {"scf.use_diis",
     [](const DefaultPair& documents) {
         return documents.written.scf.useDiis == documents.omitted.scf.useDiis;
     }},
    {"scf.trace_file",
     [](const DefaultPair& documents) {
         return documents.written.scf.traceFile == documents.omitted.scf.traceFile;
     }},
    {"scf.density_dump",
     [](const DefaultPair& documents) {
         return documents.written.scf.densityDumpFile == documents.omitted.scf.densityDumpFile;
     }},
    {"scf.checkpoint_file",
     [](const DefaultPair& documents) {
         return documents.written.scf.checkpointFile == documents.omitted.scf.checkpointFile;
     }},
    {"scf.checkpoint_converged",
     [](const DefaultPair& documents) {
         return documents.written.scf.checkpointConverged ==
                documents.omitted.scf.checkpointConverged;
     }},
    {"resources.memory_cap_gib",
     [](const DefaultPair& documents) {
         return documents.written.resources.memoryCapGiB ==
                documents.omitted.resources.memoryCapGiB;
     }},
    {"resources.thread_cap",
     [](const DefaultPair& documents) {
         return documents.written.resources.threadCap == documents.omitted.resources.threadCap;
     }},
    {"memory_instrument.enabled",
     [](const DefaultPair& documents) {
         return documents.written.memoryInstrument.enabled ==
                documents.omitted.memoryInstrument.enabled;
     }},
    {"memory_instrument.snapshot_interval_ms",
     [](const DefaultPair& documents) {
         return documents.written.memoryInstrument.snapshotIntervalMs ==
                documents.omitted.memoryInstrument.snapshotIntervalMs;
     }},
    {"memory_instrument.trace_file",
     [](const DefaultPair& documents) {
         return documents.written.memoryInstrument.traceFile ==
                documents.omitted.memoryInstrument.traceFile;
     }},
    {"diagnostics.force_disk_ri",
     [](const DefaultPair& documents) {
         return documents.written.diagnostics.forceDiskRi ==
                documents.omitted.diagnostics.forceDiskRi;
     }},
    {"symmetry.full_group",
     [](const DefaultPair& documents) {
         return documents.written.symmetry.fullGroup == documents.omitted.symmetry.fullGroup;
     }},
    {"guess.type",
     [](const DefaultPair& documents) {
         return documents.written.guess == documents.omitted.guess;
     }},
    {"guess.restart_path",
     [](const DefaultPair& documents) {
         return documents.written.guessRestartPath == documents.omitted.guessRestartPath;
     }},
    {"properties.hirshfeld",
     [](const DefaultPair& documents) {
         return documents.written.properties.hirshfeld == documents.omitted.properties.hirshfeld;
     }},
    {"properties.voronoi",
     [](const DefaultPair& documents) {
         return documents.written.properties.voronoi == documents.omitted.properties.voronoi;
     }},
    {"properties.eddb",
     [](const DefaultPair& documents) {
         return documents.written.properties.eddb == documents.omitted.properties.eddb;
     }},
    {"properties.fukui",
     [](const DefaultPair& documents) {
         return documents.written.properties.fukui == documents.omitted.properties.fukui;
     }},
    {"properties.nalewajski",
     [](const DefaultPair& documents) {
         return documents.written.properties.nalewajski == documents.omitted.properties.nalewajski;
     }},
    {"properties.density_at_nuclei",
     [](const DefaultPair& documents) {
         return documents.written.properties.densityAtNuclei ==
                documents.omitted.properties.densityAtNuclei;
     }},
    {"properties.qtaim",
     [](const DefaultPair& documents) {
         return documents.written.properties.qtaim == documents.omitted.properties.qtaim;
     }},
    {"properties.molden",
     [](const DefaultPair& documents) {
         return documents.written.properties.molden == documents.omitted.properties.molden;
     }},
    {"properties.xc_gradient",
     [](const DefaultPair& documents) {
         return documents.written.properties.xcGradient == documents.omitted.properties.xcGradient;
     }},
    // The [grid] keys' defaults live in RunGridInput (mirrored field for field
    // by the engine's own settings), so they are compared against a
    // default-constructed one rather than against RunInput - whose `grid` is an
    // absent optional until a file writes a block.
    {"grid.radial_points",
     [](const DefaultPair& documents) {
         return documents.written.grid.has_value() &&
                documents.written.grid->radialPoints == qcx::io::RunGridInput{}.radialPoints;
     }},
    {"grid.angular_points",
     [](const DefaultPair& documents) {
         return documents.written.grid.has_value() &&
                documents.written.grid->angularPoints == qcx::io::RunGridInput{}.angularPoints;
     }},
    {"grid.alpha",
     [](const DefaultPair& documents) {
         return documents.written.grid.has_value() &&
                documents.written.grid->alpha == qcx::io::RunGridInput{}.alpha;
     }},
    {"grid.radial_exponent",
     [](const DefaultPair& documents) {
         return documents.written.grid.has_value() &&
                documents.written.grid->radialExponent == qcx::io::RunGridInput{}.radialExponent;
     }},
    {"grid.trim_weight",
     [](const DefaultPair& documents) {
         return documents.written.grid.has_value() &&
                documents.written.grid->trimWeight == qcx::io::RunGridInput{}.trimWeight;
     }},
    {"grid.block_target",
     [](const DefaultPair& documents) {
         return documents.written.grid.has_value() &&
                documents.written.grid->blockTarget == qcx::io::RunGridInput{}.blockTarget;
     }},
};

// The substitute values the liveness check tries, in order: something that
// fits a bool, an int, a double, a path, and one word of every small
// vocabulary a defaulted key can hold.
constexpr const char* kAlternativeValues[] = {"true",
                                              "false",
                                              "0",
                                              "1",
                                              "2",
                                              "1.5",
                                              "6",
                                              "\"x\"",
                                              "\"bohr\"",
                                              "\"uhf\"",
                                              "\"kLoose\"",
                                              "\"gwh\"",
                                              "\"qfmm\"",
                                              "\"disk\""};

// One passthrough key's default: the value the ENGINE's own option struct
// holds, which is what an omitted key leaves in place. The schema cannot state
// these (its field is an absent optional, deliberately - a written key and an
// omitted one are different requests), so the engine's header is the authority
// the help is held to.
struct EngineDefaultClaim {
    std::string_view key;
    std::string_view authority;
    bool (*matches)(double shown);
};

const EngineDefaultClaim kEngineDefaultClaims[] = {
    {"method.screening_tolerance",
     "qcx::io::kDefaultScreeningTolerance",
     [](double shown) { return shown == qcx::io::kDefaultScreeningTolerance; }},
    {"method.theta",
     "QfmmOptions::theta",
     [](double shown) { return shown == qcx::integrals::QfmmOptions{}.theta; }},
    {"method.l_mult",
     "QfmmOptions::lMult",
     [](double shown) {
         return shown == static_cast<double>(qcx::integrals::QfmmOptions{}.lMult);
     }},
    {"method.max_leaf_size",
     "QfmmOptions::maxLeafSize",
     [](double shown) {
         return shown == static_cast<double>(qcx::integrals::QfmmOptions{}.maxLeafSize);
     }},
    // The crossover override is the one passthrough key whose engine default
    // (-1, "the built-in threshold") cannot be written: the parser admits >= 0
    // only. The help therefore shows an admissible value and says in words what
    // an omitted key keeps, and what is checkable here is that the shown value
    // is inside the admitted range.
    {"method.crossover_basis_function_count",
     "the parser admits >= 0 only",
     [](double shown) { return shown >= 0.0; }},
#ifdef QcxHasStorage
    {"method.ri_chunk_bytes",
     "DiskRiFockOptions::chunkBytes",
     [](double shown) {
         return shown == static_cast<double>(qcx::storage::DiskRiFockOptions{}.chunkBytes);
     }},
#endif
};

std::optional<double> ShownNumber(std::string_view shown) {
    double value = 0.0;
    const auto result = std::from_chars(shown.data(), shown.data() + shown.size(), value);

    if (result.ec != std::errc{} || result.ptr != shown.data() + shown.size())
    {
        return std::nullopt;
    }

    return value;
}

// ---------------------------------------------------------------------------
// The [builder] axes: the word an axis left OUT of a written block resolves to.
// ---------------------------------------------------------------------------

// The help shows a default for each axis, and that default is a fact about a
// block that does not write the axis - so the check writes a SIBLING axis (the
// block is present, the claimed axis is not) and reads the resolved word back
// through the parser's own accessor.
struct AxisClaim {
    std::string_view key; // the help row whose shown value is checked
    std::string_view siblingKey; // written instead, so the claimed axis stays absent
    std::string_view siblingValue;
};

const AxisClaim kAxisClaims[] = {
    {"builder.integral_family", "builder.storage_tier", "\"in_memory\""},
    {"builder.storage_tier", "builder.integral_family", "\"direct\""},
    {"builder.execution_backend", "builder.integral_family", "\"direct\""},
};

std::string ResolvedAxisWord(const qcx::io::RunInput& input, std::string_view key) {
    if (key == "builder.integral_family")
    {
        return input.builder.integralFamily.has_value()
                   ? std::string(qcx::io::ToString(*input.builder.integralFamily))
                   : std::string{};
    }

    if (key == "builder.storage_tier")
    {
        return input.builder.storageTier.has_value()
                   ? std::string(qcx::io::ToString(*input.builder.storageTier))
                   : std::string{};
    }

    if (key == "builder.execution_backend")
    {
        return input.builder.executionBackend.has_value()
                   ? std::string(qcx::io::ToString(*input.builder.executionBackend))
                   : std::string{};
    }

    return {};
}

std::string IssueText(const qcx::io::ValidationReport& report) {
    return report.issues.empty() ? std::string{} : report.issues.front();
}

} // namespace

// ---------------------------------------------------------------------------

TEST(RunInputHelpTest, TheHelpTextFollowsItsOwnFormat) {
    const HelpText help = ParseHelp(qcx::driver::internal::kRunHelpText);

    for (const std::string& problem : help.problems)
    {
        ADD_FAILURE() << problem;
    }

    EXPECT_FALSE(help.blocks.empty());
    EXPECT_FALSE(help.rows.empty());

    std::set<std::string> seen;

    for (const HelpRow& row : help.rows)
    {
        const std::string key = BlockKey(row);
        EXPECT_FALSE(row.shown.empty()) << key << " shows no value";
        EXPECT_FALSE(row.description.empty()) << key << " has no description";
        EXPECT_TRUE(seen.insert(key).second) << key << " is documented twice";
    }
}

// Every key the parser reads is documented, and every key the help documents
// is a key the parser reads. This is the check that fails when a key is added
// to the schema without a help row - the drift the help text had.
TEST(RunInputHelpTest, DocumentsEveryKeyTheParserReadsAndNoOther) {
    const Fixture fixture = BuildFixture();
    Report(fixture);
    ASSERT_TRUE(fixture.usable);

    const HelpText& help = fixture.help;
    const ParserKeys& keys = fixture.keys;

    ASSERT_FALSE(keys.all.empty()) << "the scan of " << kParserSource << " found no key";

    std::set<std::string> documented;
    std::set<std::string> documentedBlocks(help.blocks.begin(), help.blocks.end());

    for (const HelpRow& row : help.rows)
    {
        documented.insert(BlockKey(row));
    }

    for (const std::string& key : keys.all)
    {
        if (documented.count(key) == 0)
        {
            ADD_FAILURE() << "the parser reads " << key << " and the help does not document it";
        }
    }

    for (const std::string& key : documented)
    {
        if (keys.all.count(key) == 0)
        {
            ADD_FAILURE() << "the help documents " << key << " and the parser never reads it";
        }
    }

    std::set<std::string> parserBlocks;

    for (const std::string& key : keys.all)
    {
        parserBlocks.insert(key.substr(0, key.find('.')));
    }

    EXPECT_EQ(parserBlocks, documentedBlocks);
}

// Every value the help shows is a value the parser takes for that key - which
// is how a shown default that moved, or an example that was never legal, is
// caught without pinning the string.
TEST(RunInputHelpTest, ShowsAValueTheParserAcceptsForEveryKey) {
    const Fixture fixture = BuildFixture();
    Report(fixture);
    ASSERT_TRUE(fixture.usable);

    const HelpText& help = fixture.help;
    const Values& base = fixture.base;

    for (const HelpRow& row : help.rows)
    {
        Values values = base;
        values[BlockKey(row)] = row.shown;

        const auto parsed = qcx::io::ParseRunInput(BuildDocument(values));

        if (!parsed.has_value())
        {
            ADD_FAILURE() << BlockKey(row) << " = " << row.shown
                          << " is refused: " << parsed.error().message;
        }
    }
}

// Every closed vocabulary the parser refuses by name is spelled in the help,
// word for word. The vocabularies are not listed here: each key is probed with
// a value no vocabulary holds, and the words come off the parser's own refusal.
TEST(RunInputHelpTest, SpellsEveryClosedVocabularyTheParserRefusesBy) {
    const Fixture fixture = BuildFixture();
    Report(fixture);
    ASSERT_TRUE(fixture.usable);

    const HelpText& help = fixture.help;
    std::vector<std::string> problems;
    const std::map<std::string, std::string> vocabularies =
        ProbeVocabularies(fixture.keys, fixture.base, problems);

    for (const std::string& problem : problems)
    {
        ADD_FAILURE() << problem;
    }

    ASSERT_FALSE(vocabularies.empty()) << "no key was found to hold a closed vocabulary";

    for (const auto& [key, words] : vocabularies)
    {
        const HelpRow* row = FindRow(help, key);
        ASSERT_NE(row, nullptr) << key << " holds a vocabulary (" << words
                                << ") and the help documents no such key";

        const std::optional<std::set<std::string>> spelled = HelpWordsFor(row->description, words);

        if (!spelled.has_value())
        {
            ADD_FAILURE() << "the help's description of " << key
                          << " does not spell the parser's own words (" << words
                          << "): " << row->description;
            continue;
        }

        const std::string shown = Unquoted(row->shown);

        EXPECT_NE(spelled->count(shown), 0u)
            << key << " shows " << shown << ", which is not one of its own words (" << words << ")";
    }
}

// Where a key has a default, the help shows the parser's own default: the
// document that writes the shown value must leave the schema exactly where the
// document that omits the key left it. Each claim is also checked for
// LIVENESS - some other value must MOVE the member the claim compares - so a
// claim wired to the wrong field cannot pass on every value and defend nothing.
TEST(RunInputHelpTest, ShowsTheParsersOwnDefaultForEveryKeyThatHasOne) {
    const Fixture fixture = BuildFixture();
    Report(fixture);
    ASSERT_TRUE(fixture.usable);

    const HelpText& help = fixture.help;
    const Values& base = fixture.base;
    const qcx::io::RunInput omitted{};

    for (const DefaultClaim& claim : kDefaultClaims)
    {
        const HelpRow* row = FindRow(help, claim.key);
        ASSERT_NE(row, nullptr) << claim.key
                                << " is claimed to have a default and the help has no such row";

        Values values = base;
        values[std::string(claim.key)] = row->shown;

        const auto written = qcx::io::ParseRunInput(BuildDocument(values));

        if (!written.has_value())
        {
            ADD_FAILURE() << claim.key << " = " << row->shown
                          << " is refused: " << written.error().message;
            continue;
        }

        EXPECT_TRUE(claim.matches(DefaultPair{*written, omitted}))
            << claim.key << " = " << row->shown << " is not what an omitted key leaves in place";

        bool alive = false;

        for (const char* alternative : kAlternativeValues)
        {
            Values moved = base;
            moved[std::string(claim.key)] = alternative;

            const auto candidate = qcx::io::ParseRunInput(BuildDocument(moved));

            if (candidate.has_value() && !claim.matches(DefaultPair{*candidate, omitted}))
            {
                alive = true;
                break;
            }
        }

        EXPECT_TRUE(alive) << "no substitute value moves " << claim.key
                           << ", so its default claim compares a member the key cannot reach";
    }
}

// The keys the engine consumes are held to the ENGINE's own option struct,
// which is what an omitted key leaves in place - the schema stores an absent
// optional for them and could not state the value itself.
TEST(RunInputHelpTest, ShowsTheEnginesOwnDefaultForTheKeysTheEngineConsumes) {
    const Fixture fixture = BuildFixture();
    Report(fixture);
    ASSERT_TRUE(fixture.usable);

    const HelpText& help = fixture.help;
    const Values& base = fixture.base;

    for (const EngineDefaultClaim& claim : kEngineDefaultClaims)
    {
        const HelpRow* row = FindRow(help, claim.key);
        ASSERT_NE(row, nullptr)
            << claim.key << " is claimed to have an engine default and the help has no such row";

        const std::optional<double> shown = ShownNumber(row->shown);

        if (!shown.has_value())
        {
            ADD_FAILURE() << claim.key << " shows " << row->shown << ", which is not a number";
            continue;
        }

        EXPECT_TRUE(claim.matches(*shown))
            << claim.key << " = " << row->shown << " is not the default the engine holds ("
            << claim.authority << ")";

        Values values = base;
        values[std::string(claim.key)] = row->shown;

        const auto parsed = qcx::io::ParseRunInput(BuildDocument(values));

        if (!parsed.has_value())
        {
            ADD_FAILURE() << claim.key << " = " << row->shown
                          << " is refused: " << parsed.error().message;
        }
    }
}

// The [builder] axes show the word an axis left OUT of a written block reads
// as - the parser's own resolution, not a word copied into the help. An
// absent block's family is a different fact (the size ladder's), and the help
// states that in words rather than showing it as an axis default.
TEST(RunInputHelpTest, ResolvesTheBuilderAxesToTheWordsTheHelpShows) {
    const Fixture fixture = BuildFixture();
    Report(fixture);
    ASSERT_TRUE(fixture.usable);

    const HelpText& help = fixture.help;
    const Values& base = fixture.base;

    for (const AxisClaim& claim : kAxisClaims)
    {
        const HelpRow* row = FindRow(help, claim.key);
        ASSERT_NE(row, nullptr) << claim.key
                                << " is claimed to be an axis default and the help has no such row";

        Values values = base;
        values[std::string(claim.siblingKey)] = std::string(claim.siblingValue);

        const auto parsed = qcx::io::ParseRunInput(BuildDocument(values));

        ASSERT_TRUE(parsed.has_value()) << claim.key << ": " << parsed.error().message;

        const std::string resolved = ResolvedAxisWord(*parsed, claim.key);

        ASSERT_FALSE(resolved.empty())
            << claim.key << " did not resolve although the block writes its sibling";
        EXPECT_EQ(resolved, Unquoted(row->shown))
            << claim.key << " resolves to " << resolved << ", and the help shows " << row->shown;
    }
}

// The semantic half of the help's claims: functional, screening_tolerance and
// the whole [grid] block are read by the Kohn-Sham lanes and refused on the
// Hartree-Fock ones. That rule lives in the VALIDATOR, so it is checked
// through it - the same document is accepted with a Kohn-Sham word and refused
// with a Hartree-Fock one.
TEST(RunInputHelpTest, KeepsTheMethodSpecificKeysOnTheLanesTheHelpNames) {
    const Fixture fixture = BuildFixture();
    Report(fixture);
    ASSERT_TRUE(fixture.usable);

    const HelpText& help = fixture.help;
    const HelpRow* functional = FindRow(help, "method.functional");
    const HelpRow* radial = FindRow(help, "grid.radial_points");
    ASSERT_NE(functional, nullptr);
    ASSERT_NE(radial, nullptr);

    // The help's other values stand: these cases add one key to the required
    // set at the value the help itself shows for it. The molecule is the one
    // departure - the help's single H atom is a document the PARSER takes but
    // not one the validator does, which checks the electron count against the
    // spin multiplicity - so the parity cases use H2, two electrons and a
    // singlet.
    Values base = fixture.base;
    base["molecule.atoms"] = "[[\"H\", 0, 0, 0], [\"H\", 0, 0, 0.74]]";

    // The one key a case adds at the value the help shows for it: the method
    // word the document asks for, the key under test, and that key's shown
    // value. The three strings travel as one named value - as three adjacent
    // `std::string_view` parameters a transposed call would compile and test a
    // case other than the one it names.
    struct Probe {
        std::string_view methodWord;
        std::string_view key;
        std::string_view value;
    };

    const auto report = [&base](const Probe& probe) {
        Values values = base;
        values["method.type"] = std::string("\"") + std::string(probe.methodWord) + "\"";
        values[std::string(probe.key)] = std::string(probe.value);

        const auto parsed = qcx::io::ParseRunInput(BuildDocument(values));

        if (!parsed.has_value())
        {
            ADD_FAILURE() << "a document for " << probe.methodWord
                          << " did not parse: " << parsed.error().message;
            return qcx::io::ValidationReport{};
        }

        return qcx::io::ValidateInput(*parsed);
    };

    const qcx::io::ValidationReport rksFunctional =
        report({"rks", "method.functional", functional->shown});
    EXPECT_TRUE(rksFunctional.IsValid()) << IssueText(rksFunctional);

    const qcx::io::ValidationReport rksGrid = report({"rks", "grid.radial_points", radial->shown});
    EXPECT_TRUE(rksGrid.IsValid()) << IssueText(rksGrid);

    const qcx::io::ValidationReport rhfFunctional =
        report({"rhf", "method.functional", functional->shown});
    EXPECT_FALSE(rhfFunctional.IsValid())
        << "the validator accepts functional on rhf, and the help says it is refused";
    ASSERT_FALSE(rhfFunctional.issues.empty());
    EXPECT_NE(rhfFunctional.issues.front().find("functional"), std::string::npos)
        << IssueText(rhfFunctional);

    const qcx::io::ValidationReport rhfGrid = report({"rhf", "grid.radial_points", radial->shown});
    EXPECT_FALSE(rhfGrid.IsValid())
        << "the validator accepts a grid key on rhf, and the help says it is refused";
    ASSERT_FALSE(rhfGrid.issues.empty());
    EXPECT_NE(rhfGrid.issues.front().find("grid"), std::string::npos) << IssueText(rhfGrid);
}
