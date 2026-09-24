# Result reference


One `qcx run` produces two things: a one-page report written beside the input file, and one JSON
document. Nothing else is written unless the input asks for it — an input that names a trace, a
checkpoint, a density dump or a Molden file gets those as well.

| Destination | What arrives there |
|---|---|
| stdout | the JSON document |
| `<input>.out` | the report, beside the input file |
| stderr | diagnostics, one line per message, prefixed `qcx: ` |

The report's path is the input's own path with its extension replaced: `water.toml` produces
`water.out`, in the same directory. A path with no extension gains one rather than losing its name,
so `qcx run water` produces `water.out` as well.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | the run finished; the JSON is on stdout |
| 1 | a failed run, or an input the reader or the validator refused |
| 2 | a malformed command line |

An input the validator refuses is rejected before any computation starts: every problem it found is
printed on stderr as its own `qcx: ` line, then a closing line giving the count, and the run stops
with exit code 1. No report file is written for such a run.

A run that succeeds can still write to stderr, in two shapes. Before it starts, and when the input
asked for a cap, the hard memory cap prints one line stating the cap it applied and how it landed;
that line is not prefixed with `qcx: `, because it belongs to the cap mechanism rather than to the
run's diagnostics, and it is printed once per process. Separately, a run can warn without failing —
a builder the size ladder demoted, or a `trace_file` that recorded no rows at all. A warning carries
the `qcx: warning: ` prefix and leaves the exit code alone.

Writing the report is the last thing a successful run does, and a report that cannot be written
does not turn that run into a failure: the JSON is already on stdout, the exit code stays 0, and
the reason the file is missing goes to stderr.

### The report

```
qcx run report
------------------------------------------------------------------
molecule      H2O        charge 0     multiplicity 1
basis         sto-3g     3 atoms      7 basis functions
method        rhf        accuracy kNormal
builder       direct / lean / cpu    (size ladder)

SCF
  converged          yes          8 iterations
  total energy       -74.9631 Hartree
  |E_n - E_(n-1)|    1.7e-10
  RMS density delta  4.0e-11

Mulliken charges
  O  -0.366     H  0.183     H  0.183

timings       total 3978 ms   (scf loop 2059 ms)
```

The header states what the file asked for. The formula is built from the atom rows in the order a
chemist reads — carbon, hydrogen, then the remaining elements alphabetically — so it does not
depend on the order the file wrote its rows in. The basis name, the method word and the accuracy
preset are the file's own request; the atom and basis-function counts are the run's. The builder
line is what the run resolved, spelled as its three axes, with the vocabulary that chose it in
parentheses: the `[builder]` axes, the deprecated builder key, or the size ladder when the file
named no builder at all.

The `SCF` section states the convergence verdict with the iteration count, the total energy in
Hartree to a tenth of a millihartree, and the two residuals the convergence gate actually compared,
in scientific notation. `|E_n - E_(n-1)|` is the energy change between the last two iterations and
`RMS density delta` the root-mean-square change in the density matrix. Those two rows appear only
when the run produced them; a result that no SCF loop filled states the verdict alone rather than a
zero nobody measured.

The Mulliken section holds one entry per atom, four entries to a line, each value to three
decimals. A partial charge is the nuclear charge less the atom's gross population, so the usual
reading is "how much charge sits on this atom": a negative number is an atom that gained electron
density. When the atom symbols and the run's internal atom order can both be resolved the section
is headed `Mulliken charges` and each entry carries its symbol; when they cannot, it is headed
`Mulliken populations` and each entry is labelled `#1`, `#2`, ... by its position, rather than
pairing a symbol with a number that may belong to a different atom.

The last line gives wall times in whole milliseconds: the whole run, and the SCF loop inside it.

A section the run did not compute is left out entirely rather than printed as zeros, so what the
report does not say is as informative as what it does. The report file is opened in binary mode, so
its line endings are LF.

### The JSON document

The same result goes to stdout as one JSON document, indented two spaces, with its keys in
alphabetical order. It carries `schema_version` (currently **36**), and a reader should key on
that: the document grows as the program grows, and a consumer that ignores unknown keys keeps
working. The opening entries of one run's document, with the entries between them left out:

```json
{
  "builder_axes": {
    "execution_backend": "cpu",
    "integral_family": "direct",
    "requested_by": "ladder",
    "storage_tier": "lean"
  },
  "converged": true,
  "method": "rhf",
  "molecule_units": "angstrom",
  "schema_version": 37,
  ...
```

The document has two ways of saying "there is nothing here", and they mean different things.

Some keys are **always present** and carry `null` when the run did not compute them: the SCF keys
`energy_delta_hartree`, `rms_density_delta`, `num_removed_overlap_directions` and `spin_squared`, and
`populations` and `moments` on a run that produced no population block at all. A `null` there is
the record stating that the quantity was not measured — never a fabricated zero.

Other keys are **absent** entirely when the run had nothing to say about them: `functional` and
`xc_grid` on a Hartree-Fock run, every analysis block under `[properties]` that was not asked for,
the `symmetry` blocks when the labeling stage did not run, and `term_counters`, `certified_bound`
and `qfmm_model` outside the runs that fill them. Absent `functional` and `xc_grid` mean no density
functional was named and no grid was built; an absent `symmetry` block means the labeling stage did
not run. Read absence as the statement, not as a missing value.

| Group | Keys |
|---|---|
| identity | `schema_version`, `molecule_units`, `method`, `functional`, `xc_grid` |
| builder | `builder_axes`, `resources_resolved` |
| SCF | `converged`, `iterations`, `total_energy_hartree`, `electronic_energy_hartree`, `energy_delta_hartree`, `rms_density_delta`, `spin_squared`, `num_removed_overlap_directions` |
| symmetry | `symmetry`, `symmetry_beta`, `symmetry_blocking` |
| properties | `populations`, `moments`, `charges`, `esp`, `eddb`, `fukui`, `nalewajski`, `nocv`, `density_at_nuclei`, `qtaim`, `molden`, `xc_gradient` |
| integrity | `term_counters`, `certified_bound`, `qfmm_model` |
| timings | `timings_ms` |

`molecule_units` is always present and names the unit the geometry was read in. `method` is always
present and names the run's own physics — the same four words `[method] type` accepts. `timings_ms`
has two members, `total` and `scf_loop`, in milliseconds.

`builder_axes` is always present and holds the builder the run resolved as its three axes, plus
`requested_by` — the vocabulary the file used to ask, which is what tells the `[builder]` axes
apart from the size ladder. A file written with the deprecated `[method] fock_builder` key also
gets `legacy_spelling` and `deprecated_key` there, so the record shows which vocabulary the input
was written in. The resolution's own reasoning lives under `resources_resolved.selection`, beside
the caps that were applied (`memory_cap_gib`, `thread_cap`, `in_process_cap_applied`, and
`cap_note` when the in-process cap was not applied).

The SCF keys qualify each other. `converged` is a bare boolean and does not say what it stood on,
so `energy_delta_hartree` and `rms_density_delta` carry the residuals the gate actually compared:
on a converged run both are below the tolerances the input set. `num_removed_overlap_directions`
counts the overlap directions dropped as linearly dependent. Zero is the well-conditioned case; a
nonzero value is not a smaller calculation, since the run happened in the reduced orthonormal space
and was mapped back.


This page is linked from the project README.
