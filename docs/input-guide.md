# Input file reference


One run is described by one TOML file. A complete, working example:

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

The file is validated before any computation starts. Every problem found is reported on stderr
and the run is aborted, so a malformed input never produces a partial result.

### `[molecule]` — required

| Key | Type | Meaning |
|---|---|---|
| `charge` | integer | total electric charge |
| `multiplicity` | integer | spin multiplicity 2S+1, at least 1 |
| `atoms` | array | one row per atom: the IUPAC element symbol and x, y, z |

Coordinates are read in **Ångström** and converted to Bohr internally; the unit that was read is
echoed back in the result as `molecule_units`. One row per atom, for example
`["O", 0.0, 0.0, 0.0]`.

### `[basis]` — required

| Key | Type | Meaning |
|---|---|---|
| `orbital` | string | the orbital basis set, by bundle name |
| `aux` | string | auxiliary basis, for a density-fitted builder |

Orbital basis sets: `sto-3g`, `sto-6g`, `3-21g`, `6-31g-star`, `6-31g-dstar`, `6-311g-dstar`,
`cc-pvdz`, `cc-pvtz`, `aug-cc-pvdz`, `aug-cc-pvtz`, `def2-svp`, `def2-tzvp`, `def2-tzvpd`,
`def2-tzvppd`, `def2-qzvp`, `pcseg-1`.

Auxiliary basis sets: `cc-pvdz-rifit`, `cc-pvtz-rifit`, `cc-pvtz-jkfit`, `aug-cc-pvdz-rifit`,
`aug-cc-pvtz-rifit`, `def2-tzvpd-rifit`, `def2-qzvppd-rifit`, `def2-universal-jfit`,
`def2-universal-jkfit`.

### `[method]`

| Key | Type | Meaning |
|---|---|---|
| `type` | string | `rhf` or `uhf` — the two methods with a run path |
| `accuracy` | string | required for a run: `kLoose`, `kNormal` or `kTight` |
| `fock_builder` | string | **deprecated** — write `[builder]` instead |

The schema also carries the Kohn-Sham words `rks` and `uks`, together with their `functional` and
grid keys. They parse and validate, but no run path folds an exchange-correlation potential into
the Fock build yet, so a Kohn-Sham input is accepted and then fails with that reason.

### `[builder]`

| Key | Type | Meaning |
|---|---|---|
| `integral_family` | string | `direct`, `ri_j_link`, `ri_jk` or `qfmm` — where the integrals come from |
| `storage_tier` | string | `lean` or `in_memory` — where that family's working set lives |
| `execution_backend` | string | `cpu`, or `gpu` to run the direct family on the device |

Writing these axes together with `[method] fock_builder` is refused: they express one request.

When no builder key is written at all, a size ladder decides: the direct family's lean member up to
1000 basis functions, `ri_j_link` up to 2000, and `qfmm` above that. A tier the run cannot wire is
demoted, and the demotion is reported on stderr and in the result. Writing `fock_builder`
explicitly opts out of the ladder.

### `[scf]`

| Key | Type | Meaning |
|---|---|---|
| `max_iterations` | integer | at least 1 |
| `energy_tolerance` | float | greater than 0 |
| `density_tolerance` | float | greater than 0 |
| `use_diis` | bool | DIIS acceleration |
| `checkpoint_file` | string | path to write the last-iterate state to |
| `checkpoint_converged` | bool | also write the checkpoint on the converged exit |

### `[guess]`

| Key | Type | Meaning |
|---|---|---|
| `type` | string | `core`, `gwh`, `sad` or `restart` |
| `restart_path` | string | the checkpoint to seed from, needed when `type = "restart"` |

### `[resources]`

| Key | Type | Meaning |
|---|---|---|
| `memory_cap_gib` | float | hard process-memory ceiling in GiB; at least 0, and 0 means no cap |
| `thread_cap` | integer | OpenMP team ceiling; at least 0, and 0 means all hardware threads |

### `[symmetry]`

| Key | Type | Meaning |
|---|---|---|
| `full_group` | bool | the full-group labeling stage; `false` disables it |

### `[grid]`

The exchange-correlation integration grid, read for a Kohn-Sham input. A Hartree-Fock run builds no
grid and its result carries no `xc_grid` block:

| Key | Type | Meaning |
|---|---|---|
| `radial_points` | integer | radial points per atom |
| `angular_points` | integer | Lebedev size |
| `alpha` | float | radial mapping scale, Bohr |
| `radial_exponent` | integer | radial mapping exponent |
| `trim_weight` | float | points with a weight below this are dropped |

### `[properties]`

Each key switches one post-SCF analysis on or off. A block that is switched off is **absent** from
the result, never present and zero-filled.

| Key | Type | Meaning |
|---|---|---|
| `hirshfeld` | bool | Hirshfeld stockholder charges |
| `voronoi` | bool | Voronoi cell charges |
| `esp` | string | `chelpg`, `mk`, or empty for none |
| `eddb` | bool | delocalized-bond analysis |
| `fukui` | bool | condensed Fukui indices |
| `nalewajski` | bool | Nalewajski-Mrozek bond orders |
| `density_at_nuclei` | bool | the density at every nucleus |
| `qtaim` | bool | Bader bond critical points |
| `molden` | string | path of the Molden file to write |
| `nocv_fragments` | array | atom-index arrays, for the ETS-NOCV analysis; closed-shell only |

### `[memory_instrument]`

| Key | Type | Meaning |
|---|---|---|
| `enabled` | bool | per-term allocation attribution |
| `snapshot_interval_ms` | integer | the watchdog's row cadence, at least 1 |
| `trace_file` | string | write-through trace; empty means statistics only |

### `[diagnostics]`

| Key | Type | Meaning |
|---|---|---|
| `force_disk_ri` | bool | force the disk-backed route |


This page is linked from the project README.
