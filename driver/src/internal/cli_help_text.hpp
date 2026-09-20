#pragma once

// The qcx command line's user-facing text (driver/src/main.cpp): the usage
// block and the `qcx run` input-schema help.
//
// It lives in a header rather than inside main.cpp for one reason: the help
// is a CLAIM ABOUT THE SCHEMA, so the test that holds it to the parser has to
// read the same bytes the CLI prints. A test that read a copy would pin the
// copy (driver/tests/run_input_help_test.cpp is that test).
//
// No internal identifier codes (decision, question or document numbers) and
// no version numbers: this text ships to users, and a number that moves with
// every schema bump is a claim that drifts on its own.
//
// FORMAT, because the test parses this text rather than comparing strings to
// it:
//
//   * a block header is a line that starts with "[name]" (an optional note
//     may follow it on the same line);
//   * a key row is exactly four spaces, the key, " = ", and the value the
//     help shows for that key - its DEFAULT where it has one, an example
//     otherwise;
//   * the description follows that value on the same line or on the following
//     lines, and every continuation line is indented deeper than four spaces,
//     so a continuation can never be read as a key row - and a key row can
//     never be read as a continuation, being the only shape with an " = " at
//     four spaces;
//   * a key row's value is a spelling the parser accepts, and where the key
//     has a default it is the parser's own default (its schema struct's
//     value, or - for a key the engine consumes - the engine's own);
//   * where a key's values are a closed vocabulary, the description spells
//     the accepted words after a ":" or a ";" and separates them with " | ",
//     which is the parser's own separator in its refusal message: the test
//     compares the two word for word.
//
// The rules above are what makes "the help and the parser agree" a checkable
// statement instead of a hope; they are stated here because this file is
// where they can be broken.

namespace qcx::driver::internal {

// `qcx --help`, and the message a bare invocation prints to stderr.
inline constexpr const char* kUsageText = R"(usage: qcx <command> [options]

commands:
  run <input.toml>   run one calculation; prints the JSON result to stdout,
                     and writes a report beside the input as <input>.out
                     (qcx run --help for the input schema)

options:
  -h, --help         show this help and exit

exit codes:
  0  success
  1  failed run or invalid input (diagnostics on stderr)
  2  malformed command line
)";

// `qcx run --help`, the input schema.
inline constexpr const char* kRunHelpText = R"(usage: qcx run <input.toml>

Runs one qcx calculation from a TOML input file and prints the JSON result to
stdout, and writes a report beside the input as <input>.out. The input is
validated before any computation: every problem is reported on stderr as one
"qcx: " line, and the run stops. The exit code is 1 on a failed run or
invalid input.

input file keys (a key marked required must be written):

[molecule]
    charge = 0                      total electric charge
    multiplicity = 1                spin multiplicity 2S+1 (>= 1); must be 1
                                    on the closed-shell lanes rhf and rks
    units = "angstrom"              angstrom | bohr; the unit the atom
                                    coordinates are read in
    atoms = [["H", 0, 0, 0]]        required; one row per atom: an IUPAC symbol
                                    and x/y/z coordinates
[basis]
    orbital = "sto-3g"              required; the name of a bundled basis set
    aux = "def2-universal-jfit"     optional; the auxiliary basis, a bundled
                                    name. Read by the ri_j_link and ri_jk
                                    families; absent = the automatic choice, the
                                    matched -rifit fit where one exists and the
                                    universal fit otherwise
[method]
    type = "rhf"                    required; rhf | uhf | rks | uks
    functional = "pbe0"             optional; a name from the built-in
                                    functional registry. Required by rks and
                                    uks, refused on rhf and uhf
    screening_tolerance = 1e-10     optional; >= 0, finite; the density
                                    screening threshold of the functional's
                                    evaluation on the grid, 0.0 evaluating
                                    densely. Read by rks and uks; 1e-10 when
                                    absent
    fock_builder = "direct"         optional; DEPRECATED - accepted, never
                                    refused, never warned; write the [builder]
                                    axes instead. One of: direct | ri_j_link |
                                    ri_jk | qfmm | gpu | lean | in_memory.
                                    Absent = the size ladder (the direct
                                    family's lean member at <= 1000 basis
                                    functions, ri_j_link up to 2000, qfmm above;
                                    a tier the run cannot wire is demoted and
                                    reported on stderr). Writing it beside the
                                    [builder] axes is refused: one request, one
                                    key
    accuracy = "kNormal"            required; kLoose | kNormal | kTight
    enforce_certified_bound = false  optional; enforce the memory-budget bound
                                    over the certified fp32 lane's routing.
                                    Honoured by the direct family's machinery
                                    member only, refused on every other route
    force_certified_lane = false    optional; ask for the certified fp32 lane
                                    itself. Honoured by the direct family's
                                    machinery member only, and refused at
                                    accuracy = "kTight", whose gate admits no
                                    quartet
    theta = 0.0                     optional; finite; the multipole expansion's
                                    well-separatedness test: 0.0 (the absent
                                    key's value) is the surface ball, a positive
                                    value the centre-to-width test, a negative
                                    value the degenerate gate (every pair near
                                    field). Read by fock_builder = "qfmm";
                                    refused elsewhere
    l_mult = -1                     optional; -1..8; the multipole order: -1
                                    (the absent key's value) is the accuracy
                                    preset's order, 0..8 an explicit one. Read by
                                    fock_builder = "qfmm"; refused elsewhere
    max_leaf_size = 8               optional; >= 1; the octree's leaf-size cap;
                                    absent = 8, the engine's own cap. Read by
                                    fock_builder = "qfmm"; refused elsewhere
    crossover_basis_function_count = 0
                                    optional; >= 0; the basis-function count at
                                    which QFMM is recommended over RI-J; absent
                                    = the built-in threshold. Read by
                                    fock_builder = "qfmm"; refused elsewhere
    ri_tensor_mode = "auto"         optional; auto | disk; the RI-J ladder's
                                    rung: "disk" selects the disk rung, wired
                                    on the restricted ri_j_link leg, and an
                                    absent key leaves the rung to the engine's
                                    own estimate. The retired word "forced_disk"
                                    is refused here, with [diagnostics]
                                    force_disk_ri as its remedy
    ri_orbit_expansion = false      optional; the RI-J symmetry orbit expansion
                                    (its 3-center task grid). Read by the
                                    ri_j_link family
    eri_cache_store = ""            optional; a path; the disk-tier ERI store -
                                    a non-empty path stores the run's ERI
                                    batches there. Read on the restricted lanes'
                                    direct machinery member; refused on uhf and
                                    uks
    ri_chunk_bytes = 268435456      optional; >= 1; the disk rung's chunk size
                                    in bytes; absent = 268435456 (256 MiB), the
                                    engine's own target. Dropped, with no
                                    effect, on a family that has no disk rung
[builder]
    integral_family = "direct"      optional; direct | ri_j_link | ri_jk |
                                    qfmm; where the integrals come from. An
                                    absent [builder] block leaves the size
                                    ladder in charge; an axis left out of a
                                    written block reads direct
    storage_tier = "in_memory"      optional; lean | in_memory; where that
                                    family's working set lives - the tier a
                                    selection resolves, not a rung. "lean" is
                                    the direct family's CPU-only member; the
                                    disk rung belongs to ri_tensor_mode, and
                                    "blocked" has no key at all
    execution_backend = "cpu"       optional; cpu | gpu; "gpu" runs the direct
                                    family on the device. "gpu_split" names the
                                    unwired candidate and is refused
[scf]
    max_iterations = 100            >= 1
    energy_tolerance = 1e-8         > 0; the energy change between two
                                    iterations, in Eh
    density_tolerance = 1e-6        > 0; the density-matrix change between two
                                    iterations
    use_diis = true                 bool; DIIS acceleration
    trace_file = ""                 a path; per-iteration diagnostics appended
                                    to it, plus one line per main-SCF
                                    Fock-build call to <path>.stats
    density_dump = ""               a path; the RHF loop's per-iteration density
                                    and Fock matrices with the overlap and the
                                    core Hamiltonian (a binary stream for the
                                    density-invariant tooling). UHF ignores it
    checkpoint_file = ""            a path; saves the SCF state (the spin-summed
                                    density, the DIIS history, the previous
                                    energies) for a later restart. Written on a
                                    non-converged exit, and on convergence when
                                    checkpoint_converged = true. RHF only
    checkpoint_converged = false    bool; write the checkpoint on the converged
                                    exit too. Meaningful only with
                                    checkpoint_file
[resources]
    memory_cap_gib = 16.0           the hard process-memory ceiling in GiB;
                                    >= 0, finite (0 = no cap from the input)
    thread_cap = 0                  the OpenMP team ceiling; >= 0 (0 = every
                                    detected hardware thread, 1 = the serial
                                    path)
[memory_instrument]
    enabled = false                 bool; the per-term allocation attribution
                                    report
    snapshot_interval_ms = 250      int; the attribution watchdog's row cadence
                                    in ms; >= 1
    trace_file = ""                 a path; the write-through allocation trace
                                    (empty = statistics only, no file)
[diagnostics]
    force_disk_ri = false           bool; force the RI-J disk rung where the
                                    ladder would not pick it, by building the
                                    disk-backed builder directly. Honoured by
                                    ri_j_link under rhf; refused there on rks,
                                    uks and uhf, which cannot wire the disk
                                    store; on every other route the run keeps
                                    its own route and records the demotion with
                                    its reason
[symmetry]
    full_group = true               bool; the post-SCF full-group labeling stage
                                    (false restores the pre-stage behavior)
[grid] read by rks and uks only
    radial_points = 75              the radial points per atom; >= 1
    angular_points = 302            a Lebedev angular size: 6 | 14 | 26 | 38 |
                                    50 | 74 | 86 | 110 | 146 | 170 | 194 | 230
                                    | 266 | 302 | 350 | 434
    alpha = 0.5                     the MHL radial mapping scale, in Bohr; a
                                    finite number > 0
    radial_exponent = 2             the MHL exponent; >= 1
    trim_weight = 1e-15             the smallest |weight| kept; a finite number
                                    >= 0
    block_target = 1024             the points per spatial block; >= 1
[guess]
    type = "core"                   core | gwh | sad | restart; the initial
                                    guess. "core" is the method's own default
                                    start (on rhf that is the gwh guess); "gwh"
                                    serves rhf and uhf, "sad" needs the uhf path,
                                    and "restart" seeds from guess.restart_path
                                    and is rhf only
    restart_path = ""               a path; the checkpoint to seed from;
                                    required when type = "restart"
[properties]
    hirshfeld = false               bool; Hirshfeld stockholder charges (the
                                    analysis needs Z <= 10)
    voronoi = false                 bool; Voronoi cell charges
    esp = "chelpg"                  optional; chelpg | mk; the electrostatic
                                    potential fit. Absent = no fit
    eddb = false                    bool; EDDB delocalized-bond analysis
    fukui = false                   bool; condensed Fukui indices (the N+1 and
                                    N-1 SCFs are run for them)
    nalewajski = false              bool; Nalewajski-Mrozek bond orders (the
                                    isolated-atom fragments need Z <= 19)
    density_at_nuclei = false       bool; rho(R_A) at every nucleus, in
                                    electrons/bohr^3
    qtaim = false                   bool; Bader bond critical points and their
                                    bond paths
    molden = ""                     a path; the Molden-format file of the
                                    converged or last-iterate SCF result
    nocv_fragments = [[0, 1]]       optional; one atom-index array per fragment
                                    (the indices count the atom rows as
                                    written). ETS-NOCV, closed-shell only
)";

} // namespace qcx::driver::internal
