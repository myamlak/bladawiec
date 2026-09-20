// The qcx command-line front end: the
// `qcx run <input.toml>` invocation. The file is parsed by io, validated
// semantically, run end to end by the driver, and the JSON
// result is written to stdout; exit 0. Errors go to stderr with a nonzero
// exit: 1 for a failed run or invalid input, 2 for a malformed command
// line.

#include "internal/cli_help_text.hpp"
#include "qcx/driver/run_driver.hpp"
#include "qcx/error.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/io/result_report.hpp"
#include "qcx/io/validate_input.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {

// The CLI's user-facing text lives in a header (internal/cli_help_text.hpp)
// so that the test holding it to the parser reads the bytes this file prints.
using qcx::driver::internal::kRunHelpText;
using qcx::driver::internal::kUsageText;

int Fail(const qcx::Error& error) {
    std::fprintf(stderr, "qcx: %s\n", error.message.c_str());
    return 1;
}

// The report's path: the input's own path with its extension replaced by
// ".out", in the input's directory - `water.toml` writes `water.out`, and a
// path with no extension gains one rather than losing its name.
std::filesystem::path ReportPathFor(const std::string& inputPath) {
    std::filesystem::path reportPath(inputPath);
    reportPath.replace_extension(".out");
    return reportPath;
}

// Writes the report, or the reason it could not be written. The stream is
// binary so the report's newlines are the bytes on disk: text mode would turn
// each one into a carriage-return pair on Windows, and one line ending is one
// line ending.
qcx::Result<void> WriteReportFile(const std::filesystem::path& path, const std::string& text) {
    std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);

    if (!file)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kIOError, "cannot write the report " + path.string()});
    }

    file << text;
    file.close();

    if (file.fail())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kIOError, "cannot write the report " + path.string()});
    }

    return {};
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2)
    {
        std::fprintf(stderr, "%s", kUsageText);
        return 2;
    }

    const std::string_view command = argv[1];

    if (command == "-h" || command == "--help")
    {
        std::puts(kUsageText);
        return 0;
    }

    if (command != "run")
    {
        std::fprintf(stderr, "qcx: unknown command \"%s\"\n%s", argv[1], kUsageText);
        return 2;
    }

    if (argc == 2)
    {
        std::fprintf(stderr, "qcx: missing <input.toml>\n%s", kRunHelpText);
        return 2;
    }

    if (argc > 3)
    {
        std::fprintf(stderr, "qcx: too many arguments\n%s", kRunHelpText);
        return 2;
    }

    const std::string_view fileArg = argv[2];

    if (fileArg == "-h" || fileArg == "--help")
    {
        std::puts(kRunHelpText);
        return 0;
    }

    auto input = qcx::io::ParseRunInputFile(argv[2]);

    if (!input.has_value())
    {
        return Fail(input.error());
    }

    const auto validation = qcx::io::ValidateInput(*input);

    if (!validation.IsValid())
    {
        for (const auto& issue : validation.issues)
        {
            std::fprintf(stderr, "qcx: %s\n", issue.c_str());
        }

        std::fprintf(stderr,
                     "qcx: %zu input error(s); run aborted before any computation\n",
                     validation.issues.size());
        return 1;
    }

    auto outcome = qcx::driver::RunDriverOutcome(*input);

    if (!outcome.has_value())
    {
        return Fail(outcome.error());
    }

    std::puts(outcome->json.c_str());

    // The human-readable report goes beside the input, with no flag to ask
    // for it. It is written only for a run that produced a result (a failed
    // run returned above), and a report that cannot be written does not turn
    // that result into a failure: the run succeeded, the JSON on stdout is
    // its result, and the reason the report is missing goes to stderr.
    const std::string report =
        qcx::io::FormatRunReport(*input, outcome->result, outcome->reportFacts);
    const auto written = WriteReportFile(ReportPathFor(std::string(fileArg)), report);

    if (!written.has_value())
    {
        std::fprintf(stderr, "qcx: %s\n", written.error().message.c_str());
    }

    return 0;
}
