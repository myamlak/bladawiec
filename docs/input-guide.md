# Input file reference


One run is described by one TOML file. A complete, working example — save it as `water.toml` and
run it with `qcx run water.toml`:

```toml
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0,  0.0000, 0.0000],
    ["H", 0.0,  0.7570, 0.5870],
    ["H", 0.0, -0.7570, 0.5870],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
accuracy = "kNormal"

[scf]
max_iterations = 100
energy_tolerance = 1e-8
density_tolerance = 1e-6
use_diis = true
```

The run writes a one-page report to `water.out` and one JSON document to stdout; the result
reference describes both.

### How the file is read

Only the `[molecule]`, `[basis]` and `[method]` tables are required, and inside them only `atoms`,
`orbital`, `type` and `accuracy`. Every other table, and every other key, is optional: an omitted
key keeps the default listed below.

Values are **type-strict**. A key written with the wrong type — `max_iterations = "100"` — is an
error rather than a value the reader coerces. A key the schema does not name is ignored, and so is
a whole table it does not name, so a file may carry notes and settings for other tools. The RI disk
force is an exception to that tolerance: written outside `[diagnostics]`, it is refused by name
rather than ignored (see `[diagnostics]` below).

The parsed file is then validated before any computation starts. Every problem found is printed on
stderr, one `qcx: ` line per problem, followed by a line giving the count; the run is aborted with
exit code 1, and no partial result and no report file are produced.

### `[molecule]` — required

| Key | Type | Default | Meaning |
|---|---|---|---|
| `charge` | integer | `0` | total electric charge |
| `multiplicity` | integer | `1` | spin multiplicity 2S+1; at least 1 |
| `units` | string | `"angstrom"` | the unit the coordinates are written in: `"angstrom"` or `"bohr"` |
| `atoms` | array | — | required; a non-empty array of `[symbol, x, y, z]` rows |

Each `atoms` row is the IUPAC element symbol followed by three numbers, for example
`["O", 0.0, 0.0, 0.0]`. Coordinates are converted to Bohr as the file is read; an absent `units`
key means Angstrom, which is what every file written before the key existed means. An unknown word
is refused by name rather than read as Angstrom. The unit that was read is echoed back in the
result as `molecule_units`, so a record says how its own geometry was interpreted.

The validator checks that every symbol is a known element, that the charge does not exceed the sum
of the atomic numbers, that at least one electron is left, and that the multiplicity and the
electron count have opposite parity.

### `[basis]` — required

| Key | Type | Default | Meaning |
|---|---|---|---|
| `orbital` | string | — | required; the orbital basis set, by bundle name |
| `aux` | string | — | auxiliary basis set, for a density-fitted builder |

Orbital basis sets: `sto-3g`, `sto-6g`, `3-21g`, `6-31g-star`, `6-31g-dstar`, `6-311g-dstar`,
`cc-pvdz`, `cc-pvtz`, `aug-cc-pvdz`, `aug-cc-pvtz`, `def2-svp`, `def2-tzvp`, `def2-tzvpd`,
`def2-tzvppd`, `def2-qzvp`, `pcseg-1`.

Auxiliary basis sets: `cc-pvdz-rifit`, `cc-pvtz-rifit`, `cc-pvtz-jkfit`, `aug-cc-pvdz-rifit`,
`aug-cc-pvtz-rifit`, `def2-tzvpd-rifit`, `def2-qzvppd-rifit`, `def2-universal-jfit`,
`def2-universal-jkfit`.

The `aux` name is read by the runs that can consume one — the density-fitted builders. A run that
resolves to the direct family writes no auxiliary fit and does not read the key. Nothing here
requires an aux: a builder that needs one and has none refuses by name, and when the size ladder
reaches that builder it resolves an auxiliary basis itself.

### `[method]` — required

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | — | required; `rhf`, `uhf`, `rks` or `uks` |
| `accuracy` | string | — | required; `kLoose`, `kNormal` or `kTight` |
| `functional` | string | — | the density functional's name, for `rks` and `uks` |
| `screening_tolerance` | float | `1e-10` | finite and at least 0; `0.0` selects the dense path |
| `fock_builder` | string | — | **deprecated** — write `[builder]` instead |
| `theta` | float | — | finite; the multipole-expansion separation threshold |
| `l_mult` | integer | — | `-1` for the accuracy preset's order, `0`–`8` for an explicit one |
| `max_leaf_size` | integer | — | at least 1 |
| `crossover_basis_function_count` | integer | — | at least 0 |
| `enforce_certified_bound` | bool | `false` | enforce the certified mixed-precision budget |
| `force_certified_lane` | bool | `false` | force the certified single-precision lane on |
| `ri_tensor_mode` | string | — | `auto` or `disk`; the disk rung of the RI-J ladder |
| `ri_chunk_bytes` | integer | — | at least 1; the disk rung's chunk-size override |
| `ri_orbit_expansion` | bool | `false` | the RI-J orbit-expansion opt-in |
| `eri_cache_store` | string | — | a non-empty path requests the disk-tier ERI store |

The four keys `theta`, `l_mult`, `max_leaf_size` and `crossover_basis_function_count` are the
composed fast-multipole builder's own tuning. They are consumed by `fock_builder = "qfmm"` alone,
and a run on any other family that writes one is refused rather than run with the key dropped:
`theta` is an accuracy control, and a run that discarded it would compute at a screening the file
did not ask for.

`enforce_certified_bound` and `force_certified_lane` are two different switches, not two spellings
of one. The first asks for the budget enforcement over the single-precision lane's routing; the
second asks for the lane itself. Both are off unless the file turns them on, and both are refused
by name on a route that cannot honour them.

An `ri_tensor_mode` value of `"forced_disk"` is refused by name; the force is spelled at
`[diagnostics] force_disk_ri`.

#### Kohn-Sham keys

`rks` and `uks` are the Kohn-Sham lanes and they are wired: a Kohn-Sham run folds an
exchange-correlation potential into the Fock build. Such a run needs `functional`, and a run
without one is refused with the list of names this build ships.

Shipped functionals: `slater`, `vwn5`, `vwn3`, `pw92`, `svwn`, `spw92`, `becke88`, `pw91`, `pbe`,
`revpbe`, `rpbe`, `mpw91`, `pbesol`, `lyp`, `pbe_c`, `pw91_c`, `p86`, `b3lyp`, `pbe0`, `b3pw91`,
`mpw1pw91`, `bhandhlyp`, `b3p86`. The first seventeen are pure (non-hybrid) functionals; the last
six are hybrids, and carry a fraction of exact exchange.

`functional`, `screening_tolerance` and the whole `[grid]` block belong to this family. Writing any
of them on an `rhf` or `uhf` run is refused, never ignored: a `functional` on a Hartree-Fock run
means the author expected a density functional run, and running Hartree-Fock under that label is
the failure the refusal exists to prevent.

#### The deprecated builder key

`[method] fock_builder` is accepted and honoured, but it is deprecated: it names a whole builder in
one word, where the `[builder]` axes below keep the three questions apart. Writing it together with
any `[builder]` axis is refused — the two spellings express one request. Its words are `direct`
(also spelled `in_memory`), `lean`, `ri_j_link`, `ri_jk`, `qfmm` and `gpu`. `direct` and
`in_memory` name the same builder; `lean` names the direct family's lean member.

### `[builder]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `integral_family` | string | — | `direct`, `ri_j_link`, `ri_jk` or `qfmm` — where the integrals come from |
| `storage_tier` | string | — | `lean` or `in_memory` — where that family's working set lives |
| `execution_backend` | string | — | `cpu`, or `gpu` to run the direct family on the device |

The three axes are one request, and they are refused together with the deprecated
`[method] fock_builder` key for that reason.

A few further axis words exist but cannot be asked for at these keys, and each is refused by name
rather than quietly resolved to something else. `storage_tier = "disk"` is refused with the key
that owns the disk rung, `[method] ri_tensor_mode`, as its remedy; `storage_tier = "blocked"` names
a rung that has no request surface at all; `execution_backend = "gpu_split"` names a candidate with
no builder to execute.

**When no builder key is written at all, a size ladder decides**, by the number of orbital basis
functions:

| Basis functions | Builder |
|---|---|
| up to 1000 | the direct family's lean member |
| over 1000, up to 2000 | `ri_j_link` |
| above 2000 | `qfmm` |

The ladder is the same for all four methods, with one exception: a Kohn-Sham run whose functional
is **non-hybrid** puts `ri_j_link` ahead of the lean member, because a non-hybrid functional needs
the Coulomb part and no exchange, so the density-fitted route is the cheap one there. A hybrid
functional keeps the shared order because it needs exchange as well. The upper cutoff is a chosen
value: where a fast-multipole build actually overtakes the others is not measured here, so the
number must not be read as a derived one.

The ladder demotes rather than refusing: a tier the run cannot wire is skipped for the tier below,
and the run says so on stderr. A builder key written explicitly is used as written and is never
demoted, so an explicit `ri_j_link` without a usable auxiliary basis keeps its own refusal.

### `[scf]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `max_iterations` | integer | `100` | at least 1 |
| `energy_tolerance` | float | `1e-8` | greater than 0 |
| `density_tolerance` | float | `1e-6` | greater than 0 |
| `use_diis` | bool | `true` | DIIS acceleration |
| `trace_file` | string | `""` | path; one line per Fock build in the main SCF loop is written to `<path>.stats` |
| `density_dump` | string | `""` | path; a binary dump of the iteration sequence, read by offline analysis |
| `checkpoint_file` | string | `""` | path; the last-iterate state, written so a later run can seed from it |
| `checkpoint_converged` | bool | `false` | write that checkpoint on the converged exit too |

An empty path means the write is not requested at all, and an omitted key is the same state. These
are write-only diagnostics: none of them changes a number the run computes.

The trace rows come from the wired builder's own per-call sink, and not every family this build can
wire has one: a run whose family has none records no rows, and such a run warns on stderr rather
than leaving an empty trace that reads like a measurement.

`checkpoint_file` is written when the run exits without converging — the iteration budget spent, so
that a later run can seed from the last iterate instead of starting over. Setting
`checkpoint_converged = true` widens it to the converged exit as well. Both the write and the
`[guess] type = "restart"` read that consumes it are restricted to `rhf`; a `uhf` run that writes
`checkpoint_file` is refused, because nothing on the unrestricted side could read the file back.
A `density_dump` on a `uhf` run is ignored rather than refused.

### `[guess]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | `"core"` | `core`, `gwh`, `sad` or `restart` |
| `restart_path` | string | `""` | the checkpoint file to seed from; required when `type = "restart"` |

`core` and `gwh` are accepted by both a restricted and an unrestricted run. `sad` builds
per-element atomic fragments, which only the unrestricted path does, so it is refused on `rhf`;
the atomic-multiplicity table this version carries stops at Z = 10. `restart` seeds the restricted
solver, so it is refused on `uhf`. An absent `type` means `core`.

### `[resources]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `memory_cap_gib` | float | `16.0` | hard process-memory ceiling in GiB; finite and at least 0 |
| `thread_cap` | integer | `0` | OpenMP team ceiling; at least 0 |

`memory_cap_gib = 0` means no cap from the input, and `thread_cap = 0` means no ceiling, so every
hardware thread is available; `thread_cap = 1` restricts the run to one.

### `[symmetry]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `full_group` | bool | `true` | the full-group labeling stage; `false` disables it |

With `full_group = true` the run labels its orbitals and states the point group it found. With
`false` the stage does not run at all and the result carries no symmetry block.

### `[grid]`

The exchange-correlation integration grid, read for a Kohn-Sham input. A Hartree-Fock run builds no
grid, and writing this block on one is refused.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `radial_points` | integer | `75` | radial points per atom; at least 1 |
| `angular_points` | integer | `302` | Lebedev size; one of the sizes this build carries |
| `alpha` | float | `0.5` | radial mapping scale, in Bohr; finite and greater than 0 |
| `radial_exponent` | integer | `2` | radial mapping exponent; at least 1 |
| `trim_weight` | float | `1e-15` | finite and at least 0; points whose weight is below it are dropped |
| `block_target` | integer | `1024` | points per spatial block; at least 1 |

The Lebedev sizes this build carries are 6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266,
302, 350 and 434. An unknown size is refused with that list.

### `[properties]`

Each key switches one post-SCF analysis on or off; all of them default to off. An analysis that was
not requested is **absent** from the result rather than present as a block of zeros.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `hirshfeld` | bool | `false` | Hirshfeld stockholder charges |
| `voronoi` | bool | `false` | Voronoi cell charges |
| `esp` | string | — | `"chelpg"` or `"mk"` — the electrostatic-potential charge fit |
| `eddb` | bool | `false` | the delocalized-bond analysis |
| `fukui` | bool | `false` | condensed Fukui indices |
| `nalewajski` | bool | `false` | Nalewajski-Mrozek bond orders |
| `density_at_nuclei` | bool | `false` | the density at every nucleus |
| `qtaim` | bool | `false` | Bader bond critical points |
| `molden` | string | `""` | path; the Molden file to write |
| `nocv_fragments` | array | — | atom-index arrays; the ETS-NOCV decomposition |

Several of these cost more than the SCF they follow, because they run SCFs of their own:
`fukui` runs the N+1 and N-1 species at the same geometry, basis and settings as the parent run;
`hirshfeld` builds the promolecular atomic fragments, and `nocv_fragments` runs the fragment SCFs
the decomposition needs. All of them are off by default so that a run stays cheap until one is
asked for.

`density_at_nuclei` reports rho at each nucleus in electrons per Bohr^3; `qtaim` reports the (3,-1)
bond critical points with their bond paths, in Bohr.

For `esp`, omit the key for no fit. An empty string is refused: it is a typo for one of the two
schemes, and treating it as "not requested" would make it indistinguishable in the result from the
omitted key. For `molden`, an empty string is the opposite — it means the file was not requested —
because a path cannot be mistyped into a different valid path.

`nocv_fragments` is an array of arrays of atom indices, each group non-empty and each index
non-negative. The indices count the `atoms` rows as written in the file, starting at 0. The groups
must be disjoint and must cover every atom exactly once. The decomposition is closed-shell only,
so it is refused on a `uhf` or `uks` run.

Two analyses are limited by the element table this version carries: `hirshfeld` needs the atomic
fragments of the SAD guess, which stops at Z = 10, and `nalewajski` needs isolated-atom fragment
SCFs, which stop at Z = 19. Beyond those, the analysis refuses by name. `fukui` is refused on a
`rks` or `uks` run, because its charged species are run on the unrestricted Hartree-Fock path,
which would mix two physics under one label.

### `[memory_instrument]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | `false` | per-term allocation attribution |
| `snapshot_interval_ms` | integer | `250` | the watchdog's row cadence in milliseconds; at least 1 |
| `trace_file` | string | `""` | path; each snapshot row is flushed to it as it is taken |

An empty `trace_file` runs the instrument without the watchdog trace: the statistics are collected
and nothing is written to disk.

### `[diagnostics]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `force_disk_ri` | bool | `false` | force the disk-backed route |

A force key makes a mechanism engage that a run would otherwise reach only by its own estimate, so
that a measurement can show the mechanism ran rather than being silently swapped for the default.
It is read where the disk store can be wired. On the `ri_j_link` family with `rhf` the force is
honoured: the disk-backed store is built directly and the result reports the disk mode with
`mode_record.forced_disk = true`. On `ri_j_link` with `uhf`, `rks` or `uks` the run is refused and
the refusal names the key, because those legs have no disk route. On every other route the run keeps
its own route and records the reason the force was not taken, so a forced request is never dropped
in silence.

Those three outcomes are what the key does where it belongs. Written anywhere else — under
`[method]`, `[scf]`, `[grid]` or any other table, as a dotted `method.force_disk_ri = true`, or
beside no table at all — it never takes effect and is never ignored: the run stops before any
computation starts, and the message names both the path the key was written at and the one that is
read, `[diagnostics] force_disk_ri`. The key is refused rather than taken as an alias, because one
force keeps one spelling in one place.


This page is linked from the project README.
