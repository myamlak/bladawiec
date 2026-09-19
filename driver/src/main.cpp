// The qcx command-line front end: the
// `qcx run <input.toml>` invocation. The file is parsed by io, validated
// semantically, run end to end by the driver, and the JSON
// result is written to stdout; exit 0. Errors go to stderr with a nonzero
// exit: 1 for a failed run or invalid input, 2 for a malformed command
// line.

#include "qcx/driver/run_driver.hpp"
#include "qcx/error.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/io/validate_input.hpp"

#include <cstdio>
#include <string>
#include <string_view>

namespace {

// qcx --help and the bare-invocation message.
constexpr const char* kUsageText = R"(usage: qcx <command> [options]

commands:
  run <input.toml>   run one calculation; prints the JSON result to stdout
                     (qcx run --help for the input schema)

options:
  -h, --help         show this help and exit

exit codes:
  0  success
  1  failed run or invalid input (diagnostics on stderr)
  2  malformed command line
)";

// qcx run --help text.
constexpr const char* kRunHelpText = R"(usage: qcx run <input.toml>

Runs one qcx calculation from a TOML input file and prints the JSON result
to stdout. Diagnostics go to stderr; the exit code is 1 on a failed run.

input file keys:
  [molecule]  charge = 0                 total electric charge
              multiplicity = 1           spin multiplicity 2S+1 (>= 1)
              atoms = [["H", 0, 0, 0]]   one row per atom: IUPAC symbol and
                                         x/y/z coordinates in Angstrom
  [basis]     orbital = "sto-3g"         one of the bundled basis sets
              aux = "..."                auxiliary basis (optional; needed
                                         by ri_j_link - an explicit
                                         fock_builder = "ri_j_link", or the
                                         ladder's 1000-to-2000 tier - when
                                         the auto-selection is overridden
                                         or none is wanted)
  [method]    type = "rhf"               "rhf" | "uhf"
              fock_builder = "direct"    DEPRECATED (schema 35): accepted,
                                         never refused and never warned -
                                         write the [builder] axes below
                                         instead. Optional; absent = the
                                         size ladder: the direct family's
                                         lean member at <= 1000 basis
                                         functions, ri_j_link up to 2000
                                         (it needs an auxiliary basis; the
                                         auto-selection always resolves
                                         one, preferring the matched
                                         -rifit and falling back to the
                                         universal fit), qfmm above; a
                                         tier the run cannot wire is
                                         demoted and reported on stderr.
                                         "direct" (the budgeted
                                         machinery), both disk routes and
                                         "gpu" are explicit opt-ins; the
                                         other words are "direct" |
                                         "ri_j_link" | "ri_jk" | "qfmm" |
                                         "gpu", "lean", or its alias
                                         "in_memory" - the direct family's
                                         within-family members, admitted
                                         at ANY size
              accuracy = "kNormal"       required; "kLoose" | "kNormal" |
                                         "kTight"
  [builder]   integral_family = "direct"
                                         "direct" | "ri_j_link" | "ri_jk" |
                                         "qfmm"; where the integrals come
                                         from. Optional; absent = the size
                                         ladder, exactly as for an absent
                                         fock_builder
              storage_tier = "lean"      "lean" | "in_memory"; where that
                                         family's working set lives. The
                                         tier the SELECTION resolves, not a
                                         rung: the disk rung is
                                         [method] ri_tensor_mode's, and the
                                         blocked rung has no key at all.
                                         "lean" is the direct family's
                                         CPU-only member
              execution_backend = "cpu"  "cpu" | "gpu"; "gpu" runs the
                                         direct family on the device, and
                                         "gpu_split" names the unwired
                                         candidate and is refused. Writing
                                         these axes and [method]
                                         fock_builder together is refused:
                                         they are one request
  [scf]       max_iterations = 100      >= 1
              energy_tolerance = 1e-8    > 0
              density_tolerance = 1e-6   > 0
              use_diis = true
              checkpoint_file = ""      path; last-iterate state
                                        write (RHF): saved on the
                                        non-converged budget exit, and on
                                        convergence only when
                                        checkpoint_converged = true
              checkpoint_converged = false
                                        bool; write the checkpoint on the
                                        converged exit too
  [guess]     type = "core"              "core" | "gwh" | "sad" | "restart"
              restart_path = ""          path; the checkpoint to
                                        seed from (needed when type =
                                        "restart", RHF; "gwh" = the default
                                        RHF start)
  [resources] memory_cap_gib = 16.0     hard process-memory ceiling in
                                         GiB; >= 0 (0 = no input cap)
              thread_cap = 0            OpenMP team ceiling; >= 0 (0 =
                                         all hardware threads; 1 = the
                                         serial path)
  [symmetry]  full_group = true         default true; the post-SCF
                                        full-group labeling stage;
                                        false restores the pre-stage
                                        behavior
  [memory_instrument]
              enabled = false           bool; the per-term allocation
                                        attribution opt-in
              snapshot_interval_ms = 250 int; the watchdog's row cadence
                                        in ms; >= 1
              trace_file = ""           path; the write-through trace
                                        (empty = stats-only, no file)
  [properties] hirshfeld = true         bool; Hirshfeld stockholder charges
               voronoi = true           bool; Voronoi cell charges
               esp = "chelpg"           "chelpg" | "mk"; empty = absent
               eddb = true              bool; EDDB delocalized-bond analysis
               fukui = true             bool; condensed Fukui indices (the
                                        driver runs the N+1/N-1 SCFs)
               nalewajski = true        bool; Nalewajski-Mrozek bond orders
               density_at_nuclei = true bool; rho(R_A) at every nucleus
               qtaim = true             bool; Bader bond critical points
               molden = "out.molden"    the path of the Molden F-file
               nocv_fragments = [[0,1]] array of atom-index arrays;
                                        ETS-NOCV, closed-shell only

The input is validated semantically before any computation; every problem
found is reported on stderr, one "qcx: " line per problem.
)";

int Fail(const qcx::Error& error) {
    std::fprintf(stderr, "qcx: %s\n", error.message.c_str());
    return 1;
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

    auto result = qcx::driver::RunDriver(*input);

    if (!result.has_value())
    {
        return Fail(result.error());
    }

    std::puts(result->c_str());
    return 0;
}
