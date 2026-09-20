#include "qcx/basisset/basis_set.hpp"

#include "qcx/backend/cpu_backend.hpp"
#include "qcx/molecule/elements.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace qcx::basisset {
namespace {

constexpr std::string_view kEcpMarker = "nelec";
constexpr std::size_t kMaxShellAngularMomentum = 6; // I shells = l 6 (RI auxiliaries).

// One source line with its 1-based line number.
struct SourceLine {
    std::size_t number;
    std::string text;
};

// Builds a kInvalidArgument Error with a 1-based line-number prefix.
qcx::Error MakeError(std::size_t lineNumber, const std::string& message) {
    return qcx::Error{qcx::ErrorCode::kInvalidArgument,
                      "line " + std::to_string(lineNumber) + ": " + message};
}

// Orders basis entries by atomic number - both Merge and AppendNwchemText
// guarantee ascending Z.
constexpr auto kByAtomicNumber = [](const auto& a, const auto& b) {
    return a.atomicNumber < b.atomicNumber;
};

// The overlap of two primitives of one shell (same l, same center) under
// the module's normalization convention - the (2a/pi)^(3/4) Gaussian times
// the solid-harmonic factor 2^(l+1) sqrt(pi a^l / (2l+1)!!) of the
// integrals engine (md_defs.hpp, mirrored by tools/gen_md_reference.py) -
// in closed form:
//   S(a, b) = 2^l (4ab)^(3/4) (ab)^(l/2) / (a + b)^(l + 3/2)
// (the l = 0 case reduces to the plain s overlap (2 sqrt(ab)/(a+b))^(3/2)).
double SameShellPrimitiveOverlap(int angularMomentum, double a, double b) noexcept {
    const double ab = a * b;
    return std::pow(2.0, angularMomentum) * std::pow(4.0 * ab, 0.75) *
           std::pow(ab, 0.5 * angularMomentum) / std::pow(a + b, angularMomentum + 1.5);
}

// Renormalizes every spherical-shell contraction to unit norm under the
// convention above - every QC code does this at parse, and raw file
// coefficients carry no physical meaning. An unnormalized basis breaks the
// RI-J metric: the per-function norm spread (1e-5..1e2 for jfit auxiliaries)
// squares up in (P|Q), driving its condition number to ~1e20 and losing
// real fit directions under the RI-J eigen-inverse floor. Idempotent.
//
// Cartesian shells are left untouched: the engine's Cartesian path applies
// only the (2a/pi)^(3/4) factor (integrals/src/internal/md_batch.cpp), never
// the solid-harmonic unit norm this rescale assumes, so renormalizing a
// Cartesian shell would silently change the meaning of its file
// coefficients. Cartesians are a pass-through for now.
void NormalizeContractions(std::vector<ElementBasis>& elements) {
    for (ElementBasis& element : elements)
    {
        for (Shell& shell : element.shells)
        {
            if (!shell.isSpherical)
            {
                continue;
            }

            const std::size_t primitiveCount = shell.exponents.size();
            // The primitive-pair overlap matrix - shared by every contraction
            // row of the shell.
            std::vector<double> overlaps(primitiveCount * primitiveCount);

            for (std::size_t k = 0; k < primitiveCount; ++k)
            {
                for (std::size_t l = 0; l < primitiveCount; ++l)
                {
                    overlaps[k * primitiveCount + l] = SameShellPrimitiveOverlap(
                        shell.angularMomentum, shell.exponents[k], shell.exponents[l]);
                }
            }

            for (std::vector<double>& coefficients : shell.coefficients)
            {
                double normSquared = 0.0;

                for (std::size_t k = 0; k < primitiveCount; ++k)
                {
                    for (std::size_t l = 0; l < primitiveCount; ++l)
                    {
                        normSquared +=
                            coefficients[k] * coefficients[l] * overlaps[k * primitiveCount + l];
                    }
                }

                if (normSquared > 0.0)
                {
                    // Truly idempotent: a contraction already at unit norm
                    // (the raw BSE self-overlap deviates by ~1e-11, the
                    // post-normalization residual by ~1e-16) is left
                    // bit-unchanged. Without the guard the second pass
                    // rescales by 1/sqrt(1 +- 1e-16) and nudges every
                    // coefficient at the last ulp - the append-path promise
                    // (AppendRenormalizationIsIdempotent).
                    if (std::abs(normSquared - 1.0) < 1e-12)
                    {
                        continue;
                    }

                    const double scale = 1.0 / std::sqrt(normSquared);

                    for (double& coefficient : coefficients)
                    {
                        coefficient *= scale;
                    }
                }
            }
        }
    }
}

// Splits text into lines with their 1-based numbers; blank lines and
// comment lines ('#' or '!') are dropped.
std::vector<SourceLine> SplitLines(std::string_view text) {
    std::vector<SourceLine> lines;
    std::istringstream stream{std::string(text)};
    std::string line;
    std::size_t number = 0;

    while (std::getline(stream, line))
    {
        ++number;
        const auto first = line.find_first_not_of(" \t\r");

        if (first == std::string::npos || line[first] == '#' || line[first] == '!')
        {
            continue;
        }

        lines.push_back(SourceLine{number, line});
    }

    return lines;
}

// Parses one double strictly via std::from_chars; trailing junk is an error.
qcx::Result<double> ParseDouble(const std::string& token,
                                std::size_t lineNumber,
                                const char* what) {
    double value = 0.0;
    const auto result = std::from_chars(token.data(), token.data() + token.size(), value);

    if (result.ec != std::errc{} || result.ptr != token.data() + token.size() ||
        !std::isfinite(value))
    {
        return std::unexpected(MakeError(lineNumber, std::string("invalid ") + what));
    }

    return value;
}

// Parses one integer strictly via std::from_chars; trailing junk (including
// fractional values) is an error.
qcx::Result<int> ParseInt(const std::string& token, std::size_t lineNumber, const char* what) {
    int value = 0;
    const auto result = std::from_chars(token.data(), token.data() + token.size(), value);

    if (result.ec != std::errc{} || result.ptr != token.data() + token.size())
    {
        return std::unexpected(MakeError(lineNumber, std::string("invalid ") + what));
    }

    return value;
}

// Maps a shell-type token to angular momentum; SP maps to both s and p.
qcx::Result<std::vector<int>> ParseShellType(const std::string& token, std::size_t lineNumber) {
    if (token == "SP")
    {
        return std::vector<int>{0, 1};
    }

    const std::string names = "SPDFGHI";
    const auto position = names.find(token);

    if (token.size() != 1 || position == std::string::npos || position > kMaxShellAngularMomentum)
    {
        return std::unexpected(MakeError(lineNumber, "unknown shell type '" + token + "'"));
    }

    return std::vector<int>{static_cast<int>(position)};
}

} // namespace

const ElementBasis* BasisSet::Find(int atomicNumber) const noexcept {
    for (const auto& element : _elements)
    {
        if (element.atomicNumber == atomicNumber)
        {
            return &element;
        }
    }

    return nullptr;
}

const ElementBasis* BasisSet::Find(std::string_view symbol) const noexcept {
    for (const auto& element : _elements)
    {
        if (element.symbol == symbol)
        {
            return &element;
        }
    }

    return nullptr;
}

qcx::Result<void> BasisSet::Merge(const BasisSet& other) {
    // Two-phase: validate everything against the current state before
    // committing anything. The original code pushed elements as it went and
    // only then discovered a duplicate ECP, leaving a half-merged basis
    // behind on failure - the caller still holds the failed basis, and the
    // next successful append sees the phantom elements.
    for (const auto& element : other._elements)
    {
        if (Find(element.symbol) != nullptr)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "duplicate element '" + element.symbol + "'"});
        }
    }

    // An ECP for an element that already has an orbital basis is legal
    // (def2-style files carry both); only a second ECP for the same Z is a
    // duplicate.
    for (const auto& ecp : other._ecps)
    {
        if (std::any_of(_ecps.begin(), _ecps.end(), [&ecp](const EcpDefinition& e) {
                return e.atomicNumber == ecp.atomicNumber;
            }))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "duplicate element '" + std::to_string(ecp.atomicNumber) + "' in ECPs"});
        }
    }

    for (const auto& element : other._elements)
    {
        _elements.push_back(element);
    }

    for (const auto& ecp : other._ecps)
    {
        _ecps.push_back(ecp);
    }

    std::sort(_elements.begin(), _elements.end(), kByAtomicNumber);
    std::sort(_ecps.begin(), _ecps.end(), kByAtomicNumber);
    return {};
}

qcx::Result<void> BasisSet::AppendNwchemText(std::string_view text) {
    const auto lines = SplitLines(text);

    if (lines.empty())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument, "empty basis block"});
    }

    // Two-phase parse: everything appends to block-local copies of the live
    // lists, and only a fully successful block commits (normalize + sort, as
    // before). A failing block must leave the basis untouched - the old code
    // appended live, so an error mid-block stranded a partially parsed basis
    // and the next successful append built on it. Elements
    // already present from an earlier append must not be repeated (Merge
    // rejects the same case); within one block, repeated element headers are
    // the norm and only append shells, so only the pre-existing symbols are
    // snapshotted. ECPs are checked against the block list (the pre-existing
    // ECPs plus this block's so far): a nelec line for the same element is
    // never legitimate, within or across blocks.
    std::vector<std::string> preExistingSymbols;
    preExistingSymbols.reserve(_elements.size());

    for (const auto& elementEntry : _elements)
    {
        preExistingSymbols.push_back(elementEntry.symbol);
    }

    std::vector<ElementBasis> blockElements = _elements;
    std::vector<EcpDefinition> blockEcps = _ecps;

    // Header: BASIS "ao basis" [SPHERICAL|CARTESIAN] PRINT - the quoted name
    // contains a space, so it spans two whitespace-separated tokens.
    std::istringstream header(lines[0].text);
    std::string basisKeyword;
    std::string quotedOpen;
    std::string quotedClose;
    std::string radialCode;
    std::string printKeyword;
    header >> basisKeyword >> quotedOpen >> quotedClose >> radialCode >> printKeyword;
    bool isSpherical = true;

    if (radialCode == "SPHERICAL")
    {
        isSpherical = true;
    } else if (radialCode == "CARTESIAN")
    {
        isSpherical = false;
    } else
    {
        return std::unexpected(MakeError(
            lines[0].number, "expected 'BASIS \"ao basis\" [SPHERICAL|CARTESIAN] PRINT'"));
    }

    if (basisKeyword != "BASIS" || quotedOpen != "\"ao" || quotedClose != "basis\"" ||
        printKeyword != "PRINT")
    {
        return std::unexpected(MakeError(
            lines[0].number, "expected 'BASIS \"ao basis\" [SPHERICAL|CARTESIAN] PRINT'"));
    }

    // Body: element lines (SYMBOL SHLTYPE or SYMBOL nelec N), primitive
    // lines (2 or 3 columns), and an optional ECP block whose radial terms
    // are skipped (only the nelec stubs are kept); END terminates.
    enum class State : std::uint8_t {
        kExpectBlockStart, // Element header, ECP, or END.
        kExpectPrimitive, // Inside a shell.
        kInEcp, // Inside the ECP block; radial terms ignored.
        kAfterEnd, // Orbital basis closed; only an ECP block may follow.
        kDone, // Everything closed; only EOF may follow.
    };
    State state = State::kExpectBlockStart;
    ElementBasis* element = nullptr;
    Shell* shells[2] = {nullptr, nullptr};
    int activeShells = 0;

    for (std::size_t i = 1; i < lines.size(); ++i)
    {
        const SourceLine& line = lines[i];
        std::istringstream stream(line.text);
        std::string firstToken;
        stream >> firstToken;

        if (firstToken == "END")
        {
            // Shells end implicitly, so END may follow primitives directly;
            // it is only malformed directly after a shell header. An END
            // closes the orbital basis or the ECP block; the orbital END
            // may be followed by an ECP block (def2-style files). After the
            // ECP block's END nothing may follow: EOF only.
            if (state == State::kExpectPrimitive && shells[0] != nullptr &&
                shells[0]->exponents.empty())
            {
                return std::unexpected(
                    MakeError(line.number, "END inside a shell block - missing primitives?"));
            }

            if (state == State::kInEcp)
            {
                state = State::kDone;
                continue;
            }

            if (state == State::kAfterEnd || state == State::kDone)
            {
                return std::unexpected(MakeError(line.number, "duplicate END"));
            }

            state = State::kAfterEnd;
            continue;
        }

        if (state == State::kDone)
        {
            return std::unexpected(MakeError(line.number, "content after the closing END"));
        }

        if (state == State::kAfterEnd)
        {
            if (firstToken != "ECP")
            {
                return std::unexpected(
                    MakeError(line.number, "content after END must be an ECP block"));
            }

            state = State::kInEcp;
            continue;
        }

        if (state == State::kInEcp)
        {
            // Only SYMBOL nelec N lines are consumed; radial terms and the
            // per-shell headers of the ECP are ignored (populated once the
            // integral machinery consumes them).
            const auto* ecpElement = qcx::molecule::FindElement(firstToken);

            if (ecpElement != nullptr)
            {
                std::string marker;
                stream >> marker;

                if (marker == kEcpMarker)
                {
                    std::string countToken;
                    stream >> countToken;
                    const auto count = ParseInt(countToken, line.number, "ECP core count");

                    if (!count.has_value())
                    {
                        return std::unexpected(count.error());
                    }

                    if (*count < 1)
                    {
                        return std::unexpected(
                            MakeError(line.number, "ECP core count must be at least 1"));
                    }

                    const bool ecpExists = std::any_of(
                        blockEcps.begin(), blockEcps.end(), [&](const EcpDefinition& ecp) {
                            return ecp.atomicNumber == ecpElement->atomicNumber;
                        });

                    if (ecpExists)
                    {
                        return std::unexpected(MakeError(
                            line.number, "duplicate element '" + firstToken + "' in ECPs"));
                    }

                    blockEcps.push_back(EcpDefinition{ecpElement->atomicNumber, *count});
                }
            }

            continue;
        }

        // A new element header implicitly ends the previous shell - shells
        // run back-to-back with no separator - so element lines are valid
        // from either state, while primitive lines require an open shell.
        const bool looksNumeric = std::isdigit(static_cast<unsigned char>(firstToken[0])) ||
                                  firstToken[0] == '-' || firstToken[0] == '+' ||
                                  firstToken[0] == '.';

        if (state == State::kExpectBlockStart && looksNumeric)
        {
            return std::unexpected(
                MakeError(line.number, "primitive line where an element block was expected"));
        }

        if (!looksNumeric)
        {
            if (firstToken == "ECP")
            {
                state = State::kInEcp;
                continue;
            }

            const auto* tableEntry = qcx::molecule::FindElement(firstToken);

            if (tableEntry == nullptr)
            {
                return std::unexpected(
                    MakeError(line.number, "unknown element symbol '" + firstToken + "'"));
            }

            std::string shellType;
            stream >> shellType;

            if (shellType == kEcpMarker)
            {
                // ECP stub without a preceding ECP keyword line.
                std::string countToken;
                stream >> countToken;
                const auto count = ParseInt(countToken, line.number, "ECP core count");

                if (!count.has_value())
                {
                    return std::unexpected(count.error());
                }

                if (*count < 1)
                {
                    return std::unexpected(
                        MakeError(line.number, "ECP core count must be at least 1"));
                }

                const bool ecpExists =
                    std::any_of(blockEcps.begin(), blockEcps.end(), [&](const EcpDefinition& ecp) {
                        return ecp.atomicNumber == tableEntry->atomicNumber;
                    });

                if (ecpExists)
                {
                    return std::unexpected(
                        MakeError(line.number, "duplicate element '" + firstToken + "' in ECPs"));
                }

                blockEcps.push_back(EcpDefinition{tableEntry->atomicNumber, *count});
                state = State::kInEcp;
                continue;
            }

            auto angularMomenta = ParseShellType(shellType, line.number);

            if (!angularMomenta.has_value())
            {
                return std::unexpected(angularMomenta.error());
            }

            if (std::find(preExistingSymbols.begin(), preExistingSymbols.end(), firstToken) !=
                preExistingSymbols.end())
            {
                return std::unexpected(
                    MakeError(line.number, "duplicate element '" + firstToken + "'"));
            }
            // One SYMBOL SHLTYPE line per shell - the same element header
            // repeats for every additional shell, so each line appends.
            ElementBasis* existing = nullptr;

            for (auto& elementEntry : blockElements)
            {
                if (elementEntry.symbol == firstToken)
                {
                    existing = &elementEntry;
                }
            }

            if (existing == nullptr)
            {
                blockElements.push_back(
                    ElementBasis{tableEntry->atomicNumber, std::string(firstToken), {}});
                existing = &blockElements.back();
            }

            element = existing;

            for (const int momentum : *angularMomenta)
            {
                element->shells.push_back(Shell{momentum, isSpherical, {}, {{}}});
            }

            activeShells = static_cast<int>(angularMomenta->size());

            for (int s = 0; s < activeShells; ++s)
            {
                shells[s] = &element->shells[element->shells.size() - activeShells + s];
            }

            state = State::kExpectPrimitive;
            continue;
        }

        // Primitive line: exponent plus one coefficient per contraction.
        // SP shells carry exactly two coefficients (s and p); pure shells
        // carry k >= 1 coefficients, fixed by the first line of the shell
        // (general contractions, e.g. aug-cc-pVDZ).
        if (state == State::kExpectPrimitive)
        {
            // firstToken was consumed by the loop head - it is the exponent.
            std::vector<std::string> tokens{firstToken};
            std::string token;

            while (stream >> token)
            {
                tokens.push_back(token);
            }

            const auto exponent = ParseDouble(tokens[0], line.number, "exponent");

            if (!exponent.has_value())
            {
                return std::unexpected(exponent.error());
            }

            if (*exponent <= 0.0)
            {
                return std::unexpected(MakeError(line.number, "exponent must be positive"));
            }

            if (activeShells == 2)
            {
                if (tokens.size() != 3)
                {
                    return std::unexpected(
                        MakeError(line.number, "SP shell needs exactly three columns"));
                }

                const auto sCoefficient = ParseDouble(tokens[1], line.number, "s coefficient");
                const auto pCoefficient = ParseDouble(tokens[2], line.number, "p coefficient");

                if (!sCoefficient.has_value())
                {
                    return std::unexpected(sCoefficient.error());
                }

                if (!pCoefficient.has_value())
                {
                    return std::unexpected(pCoefficient.error());
                }

                shells[0]->exponents.push_back(*exponent);
                shells[0]->coefficients[0].push_back(*sCoefficient);
                shells[1]->exponents.push_back(*exponent);
                shells[1]->coefficients[0].push_back(*pCoefficient);
                continue;
            }

            if (shells[0]->coefficients.size() == 1 && shells[0]->coefficients[0].empty())
            {
                // First line of a pure shell: it fixes the contraction count.
                const std::size_t contractions = tokens.size() - 1;

                if (contractions == 0)
                {
                    return std::unexpected(
                        MakeError(line.number, "primitive line needs at least one coefficient"));
                }

                shells[0]->coefficients.resize(contractions);
            }

            if (tokens.size() != shells[0]->coefficients.size() + 1)
            {
                return std::unexpected(
                    MakeError(line.number, "inconsistent column count within the shell"));
            }

            shells[0]->exponents.push_back(*exponent);

            for (std::size_t c = 0; c < shells[0]->coefficients.size(); ++c)
            {
                const auto coefficient = ParseDouble(tokens[c + 1], line.number, "coefficient");

                if (!coefficient.has_value())
                {
                    return std::unexpected(coefficient.error());
                }

                shells[0]->coefficients[c].push_back(*coefficient);
            }

            continue;
        }
    }
    // EOF: only clean when the orbital END was already seen (an ECP block
    // after it must carry its own closing END) or after that block's own
    // END. Only here does the block commit to the live lists; elements and
    // ECPs are sorted by Z on exit.
    if (state == State::kAfterEnd || state == State::kDone)
    {
        _elements = std::move(blockElements);
        _ecps = std::move(blockEcps);
        NormalizeContractions(_elements);
        std::sort(_elements.begin(), _elements.end(), kByAtomicNumber);
        std::sort(_ecps.begin(), _ecps.end(), kByAtomicNumber);
        return {};
    }

    return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument, "missing END"});
}

qcx::Result<BasisSet> ParseNwchemText(std::string_view text) {
    BasisSet basis;
    const auto result = basis.AppendNwchemText(text);

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    return basis;
}

qcx::Result<BasisSet> ParseNwchemFile(std::string_view path) {
    std::ifstream stream{std::string(path)};

    if (!stream)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kIOError,
                                          "cannot read basis file '" + std::string(path) + "'"});
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    auto parsed = ParseNwchemText(buffer.str());

    if (!parsed.has_value())
    {
        return std::unexpected(
            qcx::Error{parsed.error().code, std::string(path) + ": " + parsed.error().message});
    }

    return parsed;
}

namespace {

// Collects the `.nwchem` files of a directory in deterministic order; a
// missing or non-directory path surfaces as kIOError (the no-exceptions
// Result contract).
qcx::Result<std::vector<std::string>> CollectNwchemFiles(std::string_view directory) {
    std::vector<std::string> files;

    try
    {
        for (const auto& entry : std::filesystem::directory_iterator(std::string(directory)))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".nwchem")
            {
                files.push_back(entry.path().string());
            }
        }
    } catch (const std::filesystem::filesystem_error& e)
    {
        // The exception message embeds the offending path and the OS error.
        return std::unexpected(qcx::Error{qcx::ErrorCode::kIOError, e.what()});
    }

    return files;
}

// Parses the given files in parallel and merges them into one set, in
// filename order (the callers pre-sort the list).
qcx::Result<BasisSet> ParseFilesParallel(const std::vector<std::string>& files) {
    std::vector<qcx::Result<BasisSet>> parsed(files.size());
    qcx::backend::Backend<qcx::backend::CpuTag>{}.ParallelForDynamic(
        files.size(), [&](std::size_t i) { parsed[i] = ParseNwchemFile(files[i]); });

    BasisSet merged;

    for (std::size_t i = 0; i < files.size(); ++i)
    {
        if (!parsed[i].has_value())
        {
            return std::unexpected(parsed[i].error());
        }

        const auto result = merged.Merge(*parsed[i]);

        if (!result.has_value())
        {
            return std::unexpected(result.error());
        }
    }

    return merged;
}

} // namespace

qcx::Result<BasisSet> ParseNwchemDirectory(std::string_view directory) {
    // Collect the .nwchem files first (deterministic order), then parse in
    // parallel - per-file results are independent, the merge below sorts by Z.
    auto files = CollectNwchemFiles(directory);

    if (!files.has_value())
    {
        return std::unexpected(files.error());
    }

    std::sort(files->begin(), files->end());
    return ParseFilesParallel(*files);
}

qcx::Result<BasisSet> ParseNwchemDirectoryFiltered(std::string_view directory,
                                                   std::span<const int> atomicNumbers) {
    // Resolve Z to the exact-case IUPAC symbol first: an out-of-range Z is a
    // caller error, reported deterministically before any file I/O.
    // Duplicates collapse into one symbol.
    std::vector<std::string> symbols;

    for (const int z : atomicNumbers)
    {
        const qcx::molecule::ElementData* element = qcx::molecule::FindElement(z);

        if (element == nullptr)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "cannot filter basis files by element: atomic "
                                              "number " +
                                                  std::to_string(z) + " is out of range 1..118"});
        }

        symbols.push_back(std::string(element->symbol));
    }

    std::sort(symbols.begin(), symbols.end());
    symbols.erase(std::unique(symbols.begin(), symbols.end()), symbols.end());

    auto files = CollectNwchemFiles(directory);

    if (!files.has_value())
    {
        return std::unexpected(files.error());
    }

    // Keep only the requested elements; the surviving files are then parsed
    // by the identical per-file path, so their entries - including the
    // unit-norm renormalization - are bit-identical to the unfiltered parse.
    std::vector<std::string> kept;

    for (const std::string& path : *files)
    {
        if (std::binary_search(
                symbols.begin(), symbols.end(), std::filesystem::path(path).stem().string()))
        {
            kept.push_back(path);
        }
    }

    // A requested element with no file is the same failure BuildShellPairs
    // would raise at the engine ("basis set has no entry for ..."), surfaced
    // here with the element and the directory named.
    for (const std::string& symbol : symbols)
    {
        if (std::none_of(kept.begin(), kept.end(), [&](const std::string& path) {
                return std::filesystem::path(path).stem().string() == symbol;
            }))
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "basis set has no entry for element " + symbol +
                                                  " in '" + std::string(directory) + "'"});
        }
    }

    std::sort(kept.begin(), kept.end());
    return ParseFilesParallel(kept);
}

} // namespace qcx::basisset
