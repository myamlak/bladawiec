# Result reference


### The report file

Every successful run writes a report beside its input — `water.toml` produces `water.out`. It is
meant to be read by a person:

```
qcx run report
------------------------------------------------------------------
molecule      H2O        charge 0     multiplicity 1
basis         sto-3g     3 atoms      7 basis functions
method        rhf        accuracy kNormal
builder       direct / lean / cpu    (size ladder)

SCF
  converged          yes          12 iterations
  total energy       -74.9420 Hartree
  |E_n - E_(n-1)|    3.2e-09
  RMS density delta  8.1e-07

Mulliken charges
  O  -0.397     H  0.196     H  0.201

timings       total 412 ms   (scf loop 88 ms)
```

A section the run did not compute is left out entirely rather than printed as a zero, so what the
report does not say is as informative as what it does.

### The JSON document

The same result goes to stdout as one JSON document carrying `schema_version` (currently **36**).
A key that a run had nothing to say about is **absent**, never null or zero — an absent
`functional` and `xc_grid` on a Hartree-Fock run means no density functional was named and no grid
was built, and an absent `symmetry` block means the labeling stage did not run. Read absence as the
statement, not as a missing value.

| Group | Keys |
|---|---|
| identity | `schema_version`, `molecule_units`, `method`, `functional`, `xc_grid` |
| builder | `builder_axes`, `selection`, `resources_resolved` |
| SCF | `converged`, `iterations`, `total_energy_hartree`, `electronic_energy_hartree`, `energy_delta_hartree`, `rms_density_delta`, `spin_squared`, `num_removed_overlap_directions` |
| symmetry | `symmetry`, `symmetry_beta`, `symmetry_blocking` |
| properties | `populations`, `moments`, `charges`, `esp`, `eddb`, `fukui`, `nalewajski`, `nocv`, `density_at_nuclei`, `qtaim`, `molden` |
| integrity | `term_counters`, `certified_bound`, `qfmm_model` |
| timings | `timings_ms` |

The SCF keys qualify each other. `converged` is a bare boolean and does not say what it stood on,
so `energy_delta_hartree` and `rms_density_delta` carry the residuals the gate actually compared —
on a converged run both are below the tolerances the input set. `num_removed_overlap_directions`
counts overlap directions dropped as linearly dependent: zero for a well-conditioned system, and a
nonzero value is not a smaller calculation, since the run happened in the reduced orthonormal space
and was mapped back.


This page is linked from the project README.
