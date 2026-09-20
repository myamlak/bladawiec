# qcx run — the input and output schemas

> **Staleness note (checked 2026-09-02, the QFMM schema knobs):** this
> document describes the input schema as parsed and validated by `io/` and
> the output schema as serialized by `io/src/result_json.cpp` from the
> structs in `io/include/qcx/io/result_json.hpp`. The code is the source of
> truth — if this document and the header disagree, refresh this document.

## 1. Invocation and exit codes

`qcx run <input.toml>` reads one TOML input file, runs the SCF (and any
requested property analyses), and prints one JSON document to **stdout**.
Diagnostics go to **stderr**. Exit codes: `0` on success, `1` on a failed run
or invalid input, `2` on a malformed command line. `qcx run --help` prints a
summary of the input schema.

## 2. Input schema (TOML)

The core tables are the parser's own vocabulary (checked
2026-08-26); the `[properties]` table is added below. Coordinates are
**Ångström in the file** unless `[molecule] units` says otherwise (schema 28)
— the PARSER applies that unit exactly once, at the file boundary, and every
consumer downstream is in Bohr (`Molecule::Create`'s internal convention); the
io module stores what the file says, with this one documented exception. The
resolved unit is disclosed in the record as `molecule_units` (§3.1), so a run
is self-describing.

### 2.1 Core tables

| TOML key | Field | Valid values / default |
|---|---|---|
| `[molecule] charge` | `molecule.charge` | default 0 |
| `[molecule] multiplicity` | `molecule.multiplicity` | default 1; `>= 1` (2S+1) |
| `[molecule] units` | `molecule.coordinateUnit` | Schema 28. Optional; `"angstrom"` \| `"bohr"` (literal, lowercase); **absent = `"angstrom"`**, which is the meaning of every input file written before this key existed. It names the unit the `atoms` rows are **written in**, and it is resolved at exactly ONE place — the parser (`ParseAtom` in `io/src/parse_input.cpp`) — which applies the `× kAngstromToBohr` factor for `"angstrom"` and passes `"bohr"` rows through untouched. Every `RunAtom` downstream is therefore in **Bohr**, and no consumer converts. A word outside the vocabulary (or an empty string, or a non-string) is **refused by name** with the accepted list, never silently read as Ångström: `units = "bohrs"` is the exact error this key exists to catch rather than to tolerate, because a freely-interpreted typo is how a geometry gets read at the wrong scale (2026-08-26: pyscf's Ångström default against qcx's Bohr convention; 2026-09-15: Bohr numbers written into an Ångström input file — the same mistake in the other direction, and the reason the key is explicit rather than inferred) |
| `[molecule] atoms` | `molecule.atoms` | required; rows exactly `[symbol, x, y, z]`; IUPAC symbols, exact case. The numbers are in the unit **`[molecule] units` names**, and the parser converts them once |
| `[basis] orbital` | `basis.orbital` | required; non-empty; a bundled basis dir name |
| `[basis] aux` | `basis.aux` | optional; the RI-J auxiliary basis |
| `[method] type` | `method.method` | required; `rhf` \| `uhf` \| `rks` \| `uks` |
| `[method] fock_builder` | `method.builder` | **DEPRECATED (schema 35, the owner's ruling 2026-09-17) - ACCEPTED, never refused, with the deprecated spelling recorded in the record's `builder_axes.legacy_spelling`; write the `[builder]` axes instead (§8).** Optional; absent = the direct-family default at every size (the lean Schwarz-only member at `nBasis <= 1000`, the budgeted direct machinery above; the model auto-selection is retired from the default path); `direct` \| `ri_j_link` \| `ri_jk` \| `qfmm` \| `gpu`, plus the two TIER words `lean` and `in_memory`. **This one key carries words from TWO AXES** (named by the owner's ruling 2026-09-15, because `direct` was being read as one of its own tiers): the **FAMILY** axis — where the integrals come from, the five words above — and the **TIER** axis — where that family's working set lives: `lean`, `in_memory`, `blocked`, `disk`. Two of the tier words are this key's, because both are the DIRECT family's own: `in_memory` is the tier `direct` already runs, so it is an **alias of `direct`** — same family, same tier, same behaviour, one `BuilderKind` value (`kDirect`); `lean` is the family's other tier, it names no family word, so it lands on no `BuilderKind` — `method.builder` stays EMPTY for it and the run record's `selection.builder_member` (schema 15) names the member the run wired, with both method types reaching it (the UHF seam gave the UHF path its lean arm; the per-spin adapter is the same half-mode pair the direct UHF seam assembles). The tier axis's other two words are not this key's: they name a rung inside the builder that owns them — `blocked` is integrals' `RiFullFockRung::kBlocked` (the full-RI transform accumulated in auxiliary shell ranges) and `disk` is `[method] ri_tensor_mode`'s `disk` below. **`in_memory` is a spelling of the request, not a second state.** It fills the builder slot with `kDirect`, so every consumer, every refusal and every record reads ONE value for the two words, and an in-memory-tier run is spelled `"direct"` wherever the record or a message names it (`ToString`'s canonical word: `io/include/qcx/io/run_input.hpp`; `builder`, `explicit_builder`, `builder_member` and `reasoning` all read "direct" for either spelling). `direct` is kept as the accepted alias for the same tier so that every input file, test and refusal message written before this word existed resolves exactly as it always did. **Its key-split side is the HONOUR side**, and that follows from where the word lives rather than from a preference: the key-split rule refuses a key that NAMES a mechanism on a family that cannot honour it, and a word of THIS key selects the family in the same breath — writing `in_memory` names the direct family and its in-memory tier together, so no family can receive the request without owning that tier (the same reason `lean` is never refused on a family that has the member). Nothing is refused by name and nothing is dropped: the request resolves to `kDirect` and the run wires the in-memory machinery. An unknown word is refused by name with the accepted list, which names both tier words: `unknown method.fock_builder "x" (direct \| ri_j_link \| ri_jk \| qfmm \| gpu \| lean \| in_memory)` |
| `[method] accuracy` | `method.accuracy` | required; `kLoose` \| `kNormal` \| `kTight` (literal, PascalCase) |
| `[method] enforce_certified_bound` | `method.enforceCertifiedBound` | optional; default `false`; a TOML boolean. The global budget enforcement (the owner's ruling 2026-09-11): the certified fp32 lane's routed bound sum is compared against the budget the accuracy preset derives (`PresetEnergyBudget(accuracy)` less the build's screened-out bound sum less the ladder's slack), and a build whose sum does not fit runs the fp64 lane alone for its whole quartet set. Consumed on the direct family's **machinery** members (the FastPath) only. Every other route is refused by name with `kUnimplemented` rather than quietly ignoring the key, at two places: inside `BuildFock`, where the option reaches one of the FastPath's own refusing rungs (the LightPath chunk loop, an engaged precision ladder), and at the driver's resolution point, before any builder is wired, for every route that drops the field instead — the lean member (which has no fp32 lane to enforce), the RI-J/QFMM/GPU families (which do not implement it), and both UHF legs. The resolution-point judgement is made on the resolved kind and the same within-family predicate the wiring reads, so the retargeted device-less `gpu` fallback, which lands on that machinery member, is admitted rather than refused. The outcome is reported in the `certified_bound` block (§3.9). Off by default because it is a behaviour change: it can move a build's quartets from the fp32 lane to fp64, so energies and wall times move with it |
| `[method] force_certified_lane` | `method.forceCertifiedLane` | optional; default `false`; a TOML boolean. The certified fp32 lane's **force-on** request (the owner's ruling 2026-09-13, schema 21): `true` engages the certified fp32 lane whatever the run's compute-profile probe measured; absent — or an explicit `false` — leaves the lane's DEFAULT to that probe (§3.6), which is the state every run had before this key existed. It exists because both arms of the lane's default had only ever been exercised by injecting a probe's return value in source (the host-ratio lane forced `DetectHostComputeProfile` to 31.2; the device arm did the same), so on a machine whose probe resolves OFF the lane's ON path was unreachable from any input file and its pins could only assert the OFF behaviour. `false` is **not** a force-off — there is deliberately no force-off spelling. Honoured by the direct family's **machinery** members: the RHF wiring's `directOptions`, the direct-UHF runner's coulomb/exchange halves, the RKS/UKS composition that copies those options, and the Fukui charged species that ride the same runner. Refused by name with `kUnimplemented` at ONE resolution point, judged on the resolved kind plus the same within-family predicate the wiring reads (never on a raw method word): the direct family's **lean** member (the auto-lean default at `nBasis <= 1000`, which runs no fp32 lane at all), the RI-J/QFMM/GPU families (whose own lanes do not read the request), and — on every route — `accuracy = "kTight"`, whose `MixedPrecisionThreshold` is 0.0 by construction, so the request cannot be honoured there at all. The override is VISIBLE in the record: `resources_resolved.compute_profile.certified_lane_forced` is `true` exactly when a run forced the lane, recorded BESIDE the machine's own `certified_lane_default` rather than instead of it. Independent of `enforce_certified_bound` below: one asks for the lane, the other for the budget comparison over its routing |
| `[method] theta` | `method.theta` | optional; the QFMM well-separatedness override (`QfmmOptions.theta` passthrough, the QFMM fast-multipole track): absent = the accuracy preset's theta; `0` = the preset's theta too; `< 0` = the degenerate all-near gate (nothing well separated, the bit-exact acceptance gate); `> 0` = the explicit value. **Finite only** — TOML `nan`/`inf` are rejected (each would silently bias the well-separatedness comparisons: nan/-inf classify nothing well separated, +inf everything) |
| `[method] l_mult` | `method.lMult` | optional; the QFMM multipole-order override: `-1` (the default when absent) = the accuracy preset's order; `0..8` = the explicit order (the `kQfmmMaxLMult = 8` cap — the range `QfmmJBuilder::Create` enforces) |
| `[method] max_leaf_size` | `method.maxLeafSize` | optional; the QFMM octree leaf-size cap: absent = the engine default 8; `>= 1` |
| `[method] crossover_basis_function_count` | `method.crossoverBasisFunctionCount` | optional; the QFMM-vs-RI-J crossover override: absent = the engine's `kQfmmCrossoverBasisFunctionCount` (a placeholder until a measured value lands); `>= 0` |
| `[method] ri_tensor_mode` | `method.riTensorMode` | optional; the RI-J tensor-**rung** selector (the disk-rung knob); absent = `auto`. These are TIER words, not family words (the two axes are stated at the `[method] fock_builder` row above): the key asks where the RI tensor and its working set live, never which builder family runs. `auto` = the composed in-memory ladder (the engine's budget-path estimate picks fast/light). `disk` = the explicit opt-in that engages the storage-module disk-backed builder (the ladder's LAST rung) when — and only when — the in-memory ladder refuses at the engine's Create-time estimate under the workspace budget; a fitting memory rung always rides (the knob is inert by design), and on the legacy null-budget path (`memory_cap_gib = 0`) the disk route refuses loudly — never a silent no-op. The selector selects a rung and nothing else: the FORCE is `[diagnostics] force_disk_ri` below, because a rung word re-used to mean "force" is one key naming two mechanisms (the key-split rule). The record carries the requested-vs-ran pairing (`resources_resolved.ri_tensor_mode`, schema 19), so a request on a family with no `ri_j_link` disk route is recorded rather than dropped — `not_applicable` for `disk`. **Schema 22 removed the `forced_disk` word**: it is refused by name with the key that replaced it as the remedy (§2.3) |
| `[diagnostics] force_disk_ri` | `diagnostics.forceDiskRi` | optional; default `false`; a TOML boolean (the owner's ruling 2026-09-13, schema 22). The RI disk route's **force-on** request, and the key that replaced the retired `[method] ri_tensor_mode = "forced_disk"` word: `true` makes the driver bypass the composed in-memory ladder entirely (no `RiJkFockBuilder::Create`, so no estimate-time refusal to fall back from — and, unlike `disk`, no budget path required: with `memory_cap_gib = 0` the disk builder simply runs with no budget granted) and construct the storage-module disk-backed builder directly, at any size and under any cap. It is a **measurement/diagnostic override, not a rung selector and NOT the production path**: the ladder's own semantics are untouched (`disk` stays the LAST rung of the production path, the disk-algorithms principle), and it exists so a benchmark cell can measure the disk builder itself instead of an in-memory run wearing a disk label — at benchmark sizes the ladder fits, so `disk` alone would never engage the store. `false` — explicit or absent — is **not a force-off**: it leaves every run's rung decision where it was, which is the state every run had before this key existed. Consumed on the `fock_builder = "ri_j_link"` path; on any other family the run proceeds with the resolved family's own builder and the request is **demoted with the record stating both sides** (`resources_resolved.ri_tensor_mode`), never silently dropped. The record states the override twice, for two different questions: `mode_record.forced_disk` (schema 18) is `true` exactly on this route, and `resources_resolved.ri_tensor_mode.forced` (schema 22) names this key as the request's source |

| `[method] ri_chunk_bytes` | `method.riChunkBytes` | optional; the disk rung's target chunk-size override (the `DiskRiFockOptions.chunkBytes` passthrough): absent = the disk builder's default; `>= 1`. Meaningful only with `ri_tensor_mode = "disk"` or `[diagnostics] force_disk_ri = true` (the in-memory rungs size their own chunks). Its drop on a run with no disk rung (and on `ri_j_link` when a fitting memory rung rides) is the ruled behaviour and is **disclosed, not silent**: `resources_resolved.ri_chunk_bytes` (schema 23) states the requested size, whether the hint was honoured or dropped, and why |
| `[method] ri_orbit_expansion` | `method.riOrbitExpansion` | optional; default `false`; a TOML boolean (the increment the owner ruled in on 2026-09-13). `true` makes the driver build **both** point-group reductions — the orbital basis's and the auxiliary basis's, the second being the instance the 3c grid needs because aux functions live in their own index space — and hand them to `RiJkFockBuilder`, so the screened (bra pair x aux shell) task grid is walked to ONE representative per joint group orbit and the rest of each orbit is filled by expanding the representative's block. The mechanism evaluates **3.41x fewer ERI blocks** on the grid's own fixture (264,280 representatives over 900,698 surviving cells, c8h18/def2-SVP + def2-universal-jfit) and computes the same values to 2.9e-14 absolute. **`false` and absent are the plain walk** — every cell evaluated — and are what every run did before this key existed: the engine's own permission defaults ON, so the driver's not supplying a reduction is what holds a run on the plain path, and the driver infers nothing from the group or the size. Consumed on the `fock_builder = "ri_j_link"` path only; a request on any other family is **stated, not dropped** (`resources_resolved.ri_orbit_expansion` = `not_applicable`, schema 26), and a request on a molecule whose group is trivial is stated as `inert_trivial_group` — the mechanism is then the identity. A reduction that cannot be built (an unsupported point group) refuses the run **by name** at the wiring point |
| `[scf] max_iterations` | `scf.maxIterations` | default 100; `>= 1` |
| `[scf] energy_tolerance` | `scf.energyTolerance` | default 1e-8; `> 0` |
| `[scf] density_tolerance` | `scf.densityTolerance` | default 1e-6; `> 0` |
| `[scf] use_diis` | `scf.useDiis` | default true |
| `[scf] trace_file` | `scf.traceFile` | optional; diagnostics only - empty (default) = off. When set, the SCF loop appends one `iter=` line per iteration to the file, and the driver appends one `call=` line per main-SCF Fock build to `<file>.stats`, in the row family of the wired builder: the direct family (the lean member included) and the RI-J link record their quartet counters and calibration terms, the composed QFMM family records the columns its stats-less builder can back (`call`/`wall`/`total_wall_ms`/`far_pairs` - never a fabricated quartet zero). A builder kind that backs no per-call stats at all (`gpu`, `gpu_split`) contributes no lines, and a traced run that recorded zero rows says so on stderr (`qcx: warning:`), so an empty file is never the only evidence that no timing limb came out of a run; the composed-QFMM **UHF** runner is the one traced path still without a row family. Pure write-only side-channel: nothing reads it and the run's numbers are untouched |
| `[scf] density_dump` | `scf.densityDumpFile` | optional; diagnostics only - empty (default) = off. When set, the SCF loop writes binary records (u8 tag + u64 iteration + u64 n + n*n f64 row-major; tags 0/1/2/3 = overlap / core Hamilton / density / Fock) after a `QXCDFDMP` magic + u64 matrix-dimension header, one density/Fock pair per iteration - the C12H26 discriminating-dump side channel (`ScfDensityDumpWriter`, scf_common.hpp). Pure write-only side-channel: nothing reads it and the run's numbers are untouched |
| `[scf] checkpoint_file` | `scf.checkpointFile` | optional; empty (default) = off. The last-iterate state write (**RHF only**): at the run's end the driver saves the final SCF iterate - density plus DIIS history and the previous energies - as the storage-module HDF5 checkpoint (`SaveScfCheckpoint`), the file `guess.type = "restart"` later seeds from. Written whenever the run exits without converging (the budget exit), and also on the converged exit when `checkpoint_converged = true`. A stale file at the path is removed first (the checkpoint store is append-only by contract). UHF with the key is `kUnimplemented` |
| `[scf] checkpoint_converged` | `scf.checkpointConverged` | default false; bool. Meaningful only with `checkpoint_file`: true also writes the checkpoint when the run converges |
| `[guess] type` | `guess` | default `core` when absent; `core` \| `gwh` \| `sad` \| `restart` (RHF only; seeds from the checkpoint at `restart_path`) |
| `[guess] restart_path` | `guessRestartPath` | optional; path. The checkpoint the `restart` guess seeds from; required non-empty when `type = "restart"` (parse-time error), inert with any other type (io stores what the file says) |
| `[resources] memory_cap_gib` | `resources.memoryCapGiB` | default 16.0; finite `>= 0`; 0 = no input cap |
| `[resources] thread_cap` | `resources.threadCap` | default 0; `>= 0`; 0 = all detected hardware threads; 1 = the serial path |
| `[symmetry] full_group` | `symmetry.fullGroup` | default true; bool. The post-SCF full-group labeling stage; false restores the pre-stage behavior bit-identically (the output loses the `symmetry` block, nothing else changes) |
| `[memory_instrument] enabled` | `memoryInstrument.enabled` | default false; bool. The per-term allocation-attribution opt-in: true opens the instrumented window at run entry and closes it on every exit path |
| `[memory_instrument] snapshot_interval_ms` | `memoryInstrument.snapshotIntervalMs` | default 250; `>= 1`. The watchdog's snapshot-row cadence in ms (the final row lands on close regardless) |
| `[memory_instrument] trace_file` | `memoryInstrument.traceFile` | default empty; path. The write-through trace (metadata header + snapshot rows, each row flushed so the trace survives a cap kill); empty = the stats window only, nothing written |

The `[resources]` block carries the memory and thread ceilings
that steer the algorithm choice; the block and each key inside it are
optional, but type-strict when present. `memory_cap_gib` is enforced by the
driver as a hard job-object process-memory limit (fail-closed: if the cap
cannot be applied, the run refuses), and `thread_cap` clamps the OpenMP
team before the fp64-half/fp32-full split. See
§3.6 for the enforcement and the modeled
pre-flight that refuses a configuration which cannot fit the cap.

The `[symmetry]` block carries one switch: `full_group`, the
post-SCF full-group labeling stage. The stage is a pure a-posteriori
classification — the SCF energy and state are untouched — so it runs by
default; it is the only analysis-stage switch that is **on** by default.

The `[memory_instrument]` block carries the per-term
allocation-attribution opt-in: when `enabled = true`, the driver calls
`AllocationInstrumentEnable` at run entry — after the `[resources]` caps
are applied, before any computation — and closes the window on every exit
path (success and error returns alike), so an instrumented run's
tagged-allocation traffic lands in the instrument's stats and, when
`trace_file` is set, streams write-through snapshot rows to the trace.
Fail-closed: an instrumented run whose enable fails (for example the
trace file cannot be opened) refuses at run entry. The trace metadata
carries `memory_cap_gib` in bytes (the cap `ApplyProcessCaps` enforced;
0 when uncapped) and the resolved team size; the run-identity fields stay
empty — the caller names its runs and pairs its own record to the trace.
Purely observational: the instrument never changes what the run allocates
or computes (the bit-parity pins are absolute). Empty `trace_file`
runs the window without the watchdog (stats-only, nothing written).

The `[method]` QFMM knobs (`theta` / `l_mult` / `max_leaf_size` /
`crossover_basis_function_count`, the QFMM fast-multipole track) are the
fields of `QfmmOptions` that exist in the engine but were unreachable from
TOML; each is optional and type-strict when present, parsed strictly with the
engine's own `QfmmJBuilder::Create` validation contract so a bad value is
rejected at parse time **by its schema name** — never parked silently or
deferred to a builder error that names no TOML key. The driver's `fock_builder
= "qfmm"` path is their only consumer; other runs store them inertly (`io` is
a pure schema layer — the `[basis] aux` precedent). Schema values: `theta` a
finite double as documented above (the one knob the parse tightens past the
engine's not-NaN check, the `resources.memory_cap_gib` precedent), `l_mult` an
integer in `-1..8`, `max_leaf_size` an integer `>= 1`,
`crossover_basis_function_count` an integer `>= 0` (absent = the engine
default; the `-1` sentinels are internal to `QfmmOptions`).

The three **model parameters** of the same builder — `extentModel`,
`separationMode` and the separation buffer `separationK` — are deliberately
**NOT knobs**, and there is no spelling of them at any key: they are engine
defaults (the owner's ruling of 2026-09-15 flipped the extent model and the
separation test; the 2026-09-20 change returned the separation test to the
centre-to-width form, now aimed at each preset's own angle), and a default is
the system's judgement rather than a request (§3.10 records which one ran).
`[method] theta` is a knob of the separation ANGLE, not of the test: it
overrides the angle the default test classifies with and cannot name the
other test. The distinction matters to a writer looking for a switch after
reading §3.10: the midpoint extent and the surface-ball test are reachable
**in the engine's vocabulary and in the record**, not from an input file.
Should a key ever expose one of them,
the §2.3 knob rule's side is already fixed by what such a key would NAME —
all three name a mechanism (the objects the multipole expansion is of; which
pairs are near field), exactly as `theta` names the screening — so all three
belong on the refusal side, not the size-hint side.

On the RHF path `fock_builder = "qfmm"` computes the composed RIJCOSX-style
Fock F = H + 2J_QFMM(rho) - K(rho) (the QFMM fast-multipole track): the
Coulomb half runs through the QFMM near/far split (`QfmmHfFockBuilder`,
integrals/qfmm_hf_build.*) and the exchange half through the exchange-only
direct builder at the same accuracy preset, with one core-Hamiltonian copy
subtracted back (every split result carries its own H — the double-H trap of
the direct-UHF adapter). The result is the fused direct RHF Fock up to the
QFMM far-field approximation and the split-pass summation order. Its accuracy
semantics are the honest split (restated 2026-09-04): the recorded per-preset
budgets {1e-5, 1e-7, 1e-8} Eh bound the **QFMM Coulomb half alone** (the J
half vs exact Coulomb, measured inside its budget on the committed sweep's own
rows (`benchmarks/data/qfmm_ladder_sweep_full.csv`): 1.605e-6 C12-kLoose,
3.84e-7 C24-kLoose, 1.9e-9 C24-kNormal). The composed total's exchange half is
the direct builder at the preset's fixed density gate, whose fixed-threshold
screening accumulates dropped mass with the basis — measured class 2.29e-5 at
C12H26/STO-3G (86 functions) to 1.30e-4 composed / 1.50e-4 direct at
C24H50/STO-3G (170 functions, the framework's own parse; the "194" of the code
comments is arithmetic) — so no constant total-method budget is advertised.
The composed rows are reported under the **pre-registered size-class bands**
instead (the driver's `qcx::driver::accuracy_class_report.hpp`, text-only —
the report never enters the result JSON): the run's fixture basis count places
it in a bin (0-99 / 100-199 / 200-399 / 400-799 / 800-1599, identical across
presets, fixed before the measurements and never refit), and the per-bin
aggregates print n_fixtures, the N_basis range, the max/median/95th-percentile
absolute deviation, the worst fixture, and the fitted scaling law deltaE = C *
gate * N_bf^p with p reported when the bin holds >= 3 fixtures (bins below
that are marked underpopulated and carry no fit). The band report is per-run
REPORTING, never a budget, and the discriminator band never chooses a preset
a-posteriori: the advertised ladder stays a prior claim.

On the UHF path the same builder serves both spin channels (the production
track's UHF half): `fock_builder = "qfmm"` computes F_sigma = H + J(P_alpha +
P_beta) - K(P_sigma) per spin — the Coulomb half runs once on the
half-summed density 0.5 (P_alpha + P_beta) (the QFMM far field sees the
spin-summed charge; the J loop's baked 2.0 turns 2J(0.5 P_tot) into
J(P_tot)) and the exchange half once per spin on the raw spin densities,
one H copy subtracted back per channel — the direct-UHF adapter's assembly
(`MakeDirectUhfFockBuilder`) with the QFMM builder in the coulomb
slot. The per-spin K is the same exchange-only direct half as the RHF
path's; only the J half is approximate. The combination is explicit-only
(auto-selection still wires the direct builder for UHF) and gated by the
O2 pin test against the direct-UHF pin; the other builders' per-spin
adapters stay untested (see the §2.3 combination rules).

### 2.1.1 The `scf_trace.stats` row family — column semantics

`[scf] trace_file` writes a DIAGNOSTIC side-channel beside the run record: the
`iter=` lines go to the file, the `call=` lines to `<file>.stats`. Nothing
reads either, and no run number moves because of them. The rows are
presence-driven (a column exists exactly when the wired builder backs it, so an
absent column is not a zero) and they are NOT part of the JSON output schema
below — this subsection exists because the one way they are read wrong is
systematic, and it produced a misread cost profile on 2026-09-11.

**Every span column is either a wall span or a window accumulation, and the
two are not interchangeable.** The rule:

- `wall` is CUMULATIVE seconds since the stream opened. A per-call wall is the
  DIFFERENCE between consecutive rows; the first row's own value is its own
  call's wall (the epoch opens at the first call).
- At `thread_cap = 1` (one row window) every `*_wall_ms` column is a true WALL
  SPAN, and `eri + contract <= wall`, with the remainder the screen + flush +
  reduce residual.
- At `thread_cap > 1` the lean builder splits the row space into
  `window_count` windows and sums each window's OWN spans into the row's
  columns after the join. The windows' intervals OVERLAP, so the columns are
  window-ACCUMULATED totals — per-thread CPU-style sums whose unit is pooled
  thread-work, not wall time. They can exceed `wall` by up to the team size,
  and their shares of `wall` are meaningless there. The one wall-comparable
  quantity in such a row is `wall` itself, plus the per-window spread
  columns.

The pooled thread-work a k > 1 row's shares must be read against is
`window_count x window_wall_mean_ms`, not `wall`. Reading the accumulated
columns against `wall` is exactly the misread named above.

**The lean row's columns.** `total_wall_ms` is one `steady_clock` span around
the whole call on the calling thread — wall-comparable at every k, unlike
everything after it. `eri_wall_ms` is the flush's block-compute span (pair
data + assembly + tile setup + the batch dispatch) and `eri_prep_wall_ms` its
pre-dispatch part, so `eri - prep` is the KERNEL time proper;
`eri_prep_pair_ms` / `eri_prep_assemble_ms` / `eri_prep_tile_ms` are the prep
span's three named parts, disjoint by contract, so they sum to AT MOST the prep
span; `contract_wall_ms` is the per-quartet contraction loop. The counters
`batch_count` (assembled class batches), `pair_builds` (contracted pair
transforms the flush's pair pass actually BUILT), and the quartet column
`nq64` denominate them.

**The kernel-span probe (schema-visible since 2026-09-12).**
Three disjoint sub-spans subdivide the kernel time and four counters
denominate them. All seven are lean-only — the machinery paths leave them at
zero, and the columns are omitted from a row whose builder does not back them:

| column | meaning |
|---|---|
| `kernel_vrr_wall_ms` | The VRR/Boys phase: one pass-1 group's one ket primitive pair — the pq/acc block zeroing and the `RunVrrQuadruple` recurrence loop of every task of that group |
| `kernel_ket_wall_ms` | The ket transform: the group's one batched strided GEMM, from the close of the VRR phase to its return |
| `kernel_bra_wall_ms` | The bra transform: the whole pass-2 task loop — the per-task transform, its packed-block write and the loop's own per-task bookkeeping |
| `kernel_vrr_quads` | The VRR phase's work denominator: `RunVrrQuadruple` calls, summed over every task, ket primitive pair and bra primitive pair |
| `kernel_gate_seam_calls` | Transform calls that MISSED the micro-gate shape test and reached the linalg batched seam (ket transform calls are `kernel_prim_passes`, bra transform calls `nq64`, so the eligible share is derivable) |
| `kernel_groups` | The pass-1 group census |
| `kernel_prim_passes` | The `(group, ket primitive pair)` iteration count — exactly the ket transform call count |

The three spans are sub-spans of the kernel time, so
`kernel_vrr_wall_ms + kernel_ket_wall_ms + kernel_bra_wall_ms <= eri_wall_ms -
eri_prep_wall_ms` at every k. The inequality is deliberately not an equality:
the remainder — the per-batch scratch layout walk, `scratch.assign`, the group
splitting, the loop increments and the pass-2 clock opens — belongs to no
phase. Above k = 1 these columns follow the window-accumulated convention
above, like every other span in the family.

### 2.2 The `[properties]` block

The block is optional; when present, every analysis key inside it is
optional and defaults to **off**, but is type-strict when present (a wrong
type is a parse error). The population/moment block has no switch
— it always runs.

| TOML key | Valid values / behavior |
|---|---|
| `[properties] hirshfeld` | bool; Hirshfeld stockholder charges. Needs SAD fragment densities; limited to Z <= 10 (see §5) |
| `[properties] voronoi` | bool; Voronoi cell charges |
| `[properties] esp` | string, `"chelpg"` \| `"mk"`; an empty string counts as absent |
| `[properties] eddb` | bool; EDDB delocalized-bond analysis |
| `[properties] fukui` | bool; condensed Fukui indices (driver runs the N+1 / N-1 SCFs) |
| `[properties] nalewajski` | bool; Nalewajski-Mrozek bond orders; limited to Z <= 19 (see §5) |
| `[properties] density_at_nuclei` | bool; the electron density at every nucleus, a point evaluation with no quadrature |
| `[properties] qtaim` | bool; the Bader QTAIM bond critical points: the (3,-1) critical points of rho with their bond paths, ellipticity and laplacian; failed searches are reported in the output, never a run failure |
| `[properties] molden` | string; the path of the Molden-format F-file written from the SCF result: the [Molden Format] [Atoms] (AU) [5D] [7F] [GTO] [MO] sections, RHF one spin block, UHF alpha then beta; an empty string counts as absent, and a file that cannot be written fails the run |
| `[properties] nocv_fragments` | array of non-empty arrays of non-negative atom indices (0-based, file row order); ETS-NOCV, closed-shell only (see §5) |

### 2.3 Validation and combination rules

**Parse-time (io, fail-fast, one error, exit 1):** TOML syntax/type errors,
missing required keys, the enum vocabularies above (each `Parse*` reports
the valid-value list), `[scf]` numeric ranges, atoms non-empty, atom rows of
shape `[symbol, x, y, z]`, non-empty symbols, non-negative
`nocv_fragments` indices, the `[resources]` ranges, and the `[grid]` block's six keys — each optional, each defaulting to the engine's own compile-time value so an absent block builds exactly the grid every run built before the block existed (`radial_points` 75, `angular_points` 302, `alpha` 0.5, `radial_exponent` 2, `trim_weight` 1e-15, `block_target` 1024; `RunGridInput`, schema 34), and each refusing a nonsense value BY NAME rather than clamping: `radial_points`/`radial_exponent`/`block_target` `>= 1`, `alpha` a finite number `> 0`, `trim_weight` a finite number `>= 0`, `angular_points` one of the shipped Lebedev sizes (the refusal lists them). The block itself is refused on an `rhf`/`uhf` run, which integrates no density functional and so builds no XC grid — the XcKeyPolicy rule `method.functional` and `method.screening_tolerance` already follow, and the one the record's `xc_grid` could otherwise only have reported as a silent absence. Errors of this class
read like:

```
unknown method.fock_builder "x" (direct | ri_j_link | ri_jk | qfmm | gpu | lean | in_memory)
molecule.atoms must be a non-empty array
multiplicity must be at least 1
unknown molecule.units "bohrs" (angstrom | bohr)
each properties.nocv_fragments group must be a non-empty array of atom indices
resources.memory_cap_gib must be a finite number >= 0
resources.thread_cap must be >= 0
memory_instrument.snapshot_interval_ms must be >= 1
method.theta must be a finite number
method.l_mult must be in -1..8
method.max_leaf_size must be >= 1
method.crossover_basis_function_count must be >= 0
unknown method.ri_tensor_mode "ram" (auto | disk)
method.ri_tensor_mode does not take "forced_disk": the forced disk mode is not a rung selection and is no longer spelled here. Set [diagnostics] force_disk_ri = true instead (the accepted rung words are auto | disk)
"diagnostics.force_disk_ri" must be a bool
method.ri_chunk_bytes must be >= 1
method.ri_orbit_expansion must be a bool
guess.type = "restart" needs a non-empty guess.restart_path
```

**Semantic validation (C1):** after parsing, every atom symbol is resolved
against the element table; unknown symbols, impossible charges, and
multiplicity/electron-count parity mismatches are collected and reported
together before any computation:

```
molecule.atoms[0]: unknown element symbol "Xx"
molecule: charge 5 exceeds the sum of atomic numbers 2
molecule: no electrons (charge equals the sum of atomic numbers)
molecule: multiplicity 2 is incompatible with 2 electrons (multiplicity and
  electron count must have opposite parity)
basis.orbital: <unresolved-basis issue>
```

**Combination rules (driver, before the SCF, `kUnimplemented`):** the
schema's not-yet-wired combinations are rejected up front so a silent
substitution can never run the wrong physics:

```
fock_builder = "ri_jk" is wired for RHF (integrals
  RiFullFockBuilder); the approximation it introduces is DISCLOSED in the run record,
  never silent
UHF is wired with fock_builder = "direct", "qfmm" and "ri_j_link" in v1 (ri_jk, gpu and
  gpu_split have no per-spin adapter; the ri_j_link disk rungs are refused separately, by
  name) - "ri_j_link" ADDED 2026-09-16, which this line predated
the SAD guess needs the per-element atomic fragment inputs only the UHF path builds in
  v1; RHF supports guess core, gwh and restart (gwh is also the RHF loop's default
  start)
guess restart seeds the RHF solver in v1 (the only solver with a restart-state option);
  UHF supports guess core, gwh and sad
scf.checkpoint_file saves the RHF last-iterate state in v1 (the restart read path is
  RHF-only); UHF runs cannot write a checkpoint nothing could consume
the ETS-NOCV analysis is closed-shell in v1; nocv_fragments cannot be requested on a UHF run
nocv_fragments atom index 7 is out of range (the file lists 2 atoms)
method.ri_tensor_mode = "disk" needs the budget path: memory_cap_gib = 0 keeps the legacy
  null-budget path, whose engine ladder never refuses - the disk rung (the ladder's last
  rung) can only engage on that refusal. Set resources.memory_cap_gib above the base term
```

The restart read adds one content-dependent check (driver, before the
SCF, `kInvalidArgument`): the checkpoint at `guess.restart_path` is
verified against the run's molecule and basis (storage's fingerprint
group) and refused when it is not an RHF store — a UHF store has empty
`density` on load, and seeding would silently fall back to the GWH
start, making "restart" not a restart:

```
the checkpoint at guess.restart_path stores a UHF state; guess restart seeds the RHF
  solver in v1
```

An unreadable or corrupt file keeps storage's own error code
(`kIOError`), with the schema key named: `guess.restart_path:
<storage message>`.

## 3. Output schema (JSON)

One JSON document per run, pretty-printed with two-space indent (nlohmann's
default object type emits keys in sorted order — key order is **not** part
of the contract). Every document carries `schema_version`, an integer
bumped whenever a consumer must notice a change (§7).

### 3.1 Always-present members

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | The schema version; currently 37 (`RunResult::kSchemaVersion`) |
| `molecule_units` | string | Schema 28. The unit the run's geometry was **read under** — the RESOLVED `[molecule] units` value, `"angstrom"` or `"bohr"`, never a request left for a consumer to apply. **Always present**, and `"angstrom"` is the value an absent key resolves to (the parser's default is a judgement, and the disclosure rule makes a judgement stateable). It exists so a record is self-describing on the one convention that has produced a wasted hunt twice in this project (2026-08-26, pyscf's Ångström default; 2026-09-15, Bohr numbers in an Ångström input file): a reader holding the JSON alone can now tell at what scale the geometry was read without consulting the input file. It describes the INPUT, not the numbers — the internal geometry is Bohr under either value |
| `converged` | bool | True when a convergence criterion fired |
| `iterations` | int | Iterations spent; the budget when not converged |
| `energy_delta_hartree` | number \| null | The achieved \|E_n − E_{n−1}\| **of the returned iterate** — the gate's own energy operand, recorded to qualify a bare `converged`. **null when no SCF loop filled it** (never a fabricated 0.0). Both gate legs are live on both paths, so a converged result has this below `[scf] energy_tolerance` by construction. It was **not** bounded on RHF until 2026-09-12 — the energy leg was inert there (closed), so a default-gate run carried a value orders of magnitude above the tolerance it was given. Cached RHF results from before that date are not comparable with this one (§7) |
| `rms_density_delta` | number \| null | The RMS density change the gate compared (Frobenius/n); same null rule; below `[scf] density_tolerance` by construction. On RHF **before 2026-09-12** (when the energy-leg deferral closed) it was the leg that decided the stop while the energy leg was inert; since then both legs gate, so the stop requires this one AND `energy_delta_hartree` (§7's `16 → 17` correction) |
| `num_removed_overlap_directions` | int \| null | The **linear-dependence removal**'s count (schema 27, the disclosure rule): how many overlap directions the orthogonalizer removed because they fell below `kOverlapEigenvalueFloorTolerance` (1e-8 relative) times the largest overlap eigenvalue. A **system** property — the basis set and the molecule — not a per-method switch, so RHF and UHF report the SAME count for the same inputs. **null when no SCF loop filled it**, never a fabricated 0.0; 0 is the well-conditioned answer. A non-zero value is **not** a smaller calculation: the run happened in the reduced orthonormal space and was mapped back, so `density`, `fock` and the energies are all in the full AO dimension. It exists because a silent truncation would be indistinguishable from a correct run, and a large diffuse basis is exactly where the removal is expected rather than exceptional. It is also the disclosure of **what the removal disengages**, and the only one: a nonzero count means the blocked diagonalization fell back to the plain `n`-dimensional solve and the `symmetry`/`symmetry_beta` blocks are absent **by construction** (§3.7) — a non-C1 run that trips the floor completes and returns the full-space answer instead of refusing the run for non-square coefficients |
| `total_energy_hartree` | number | Electronic plus nuclear repulsion |
| `electronic_energy_hartree` | number | 1/2 Tr[D (H + F)] |
| `spin_squared` | number \| null | UHF spin-contamination diagnostic; **null on RHF** (not applicable — the key is present with value null, never omitted, never a fabricated 0.0) |
| `populations` | object \| null | The population block (§3.2); null only when never computed |
| `moments` | object \| null | The electric moments (§3.3); same gating |
| `timings_ms` | object | `{ "total": <whole-run wall ms>, "scf_loop": <the Run*Scf call ms> }` |
| `resources_resolved` | object | The caps the run was actually executed under (§3.6) |

`term_counters` (an instrumented ri_j run only) is documented in §3.8,
`certified_bound` (the direct family's machinery members only) in §3.9,
`qfmm_model` (a run that wired the composed-QFMM builder) in §3.10,
`symmetry`/`symmetry_beta` in §3.7 — all **absent** (never null) when
their gating conditions do not hold.

In v1 the driver computes the population/moment block for **every completed run** —
converged or last-iterate density — so real output carries the
`populations`/`moments` objects; `null` is the not-computed state (covered
by the serializer's unit tests, not reachable from a completed run).

### 3.2 `populations`

Per-atom values in the molecule's canonical atom order, in **electrons**.

```json
"populations": {
  "mulliken": { "alpha": [...], "beta": [...], "total": [...], "spin": [...] },
  "lowdin":   { "alpha": [...], "beta": [...], "total": [...], "spin": [...] },
  "mayer": {
    "bond_orders":    [[...], [...]],   // nAtoms x nAtoms, symmetric, zero diagonal
    "free_valences":  [...],            // zero for RHF by construction
    "total_valences": [...]
  },
  "gopinathan_jug": {
    "bond_orders":    [[...], [...]]    // nAtoms x nAtoms, symmetric, diagonal included
  }
}
```

**RHF convention:** alpha == beta == D/2, so the spin arrays are identically
zero **by construction** — a consumer unfamiliar with the codebase could
misread an all-zero spin-population array as a bug; it is the documented,
correct RHF behavior.

### 3.3 `moments`

```json
"moments": {
  "dipole":     [0.0, 0.0, 0.0],                 // (d_x, d_y, d_z), e*a0
  "quadrupole": [[0.0, 0.0, 0.0],                // 3x3 symmetric, traceless
                 [0.0, 0.0, 0.0],                // Theta = 3M - Tr(M) I, e*a0^2
                 [0.0, 0.0, 0.0]]
}
```

The origin convention is the run input's coordinate origin **as given** — no
automatic recentering (recenter the molecule in the input file first if a
center-of-mass dipole is wanted).

### 3.4 Opt-in blocks

Each opt-in analysis block is emitted **only when the run requested the analysis**;
an unrequested block's key is **absent from the document** (not null — see
§3.5). This is the one asymmetry with `spin_squared`/`populations`/`moments`,
which are always present and use null.

| Key | Members | Units |
|---|---|---|
| `charges` | `hirshfeld`: array \| null, `voronoi`: array \| null — per atom, each flag-gated independently | electrons |
| `esp` | `charges` (per atom), `rms_error`, `point_count` | electrons; hartree; count |
| `eddb` | `total_population`, `atomic_populations`, `nobd_occupations` (descending), `central_atom_count`, `two_center_orbital_count` | electrons; counts |
| `fukui` | `nucleophilic` (f_A+), `electrophilic` (f_A-), `radical` (f_A0) — per atom, each vector sums to one | electrons |
| `nalewajski` | `bond_orders` (B(a,b)), `diatomic_covalent` (v_ab), `atomic_ionic_valence` (v_i_a), `atomic_covalent_valence` (v_c_a), `total_valence` — matrices symmetric, zero diagonal | electrons |
| `nocv` | `electrostatic`, `pauli`, `orbital`, `orbital_unrestricted`, `binding_energy` (= E_mol - sum E_frag = elstat + pauli + orb), `fragment_energies`, `orbital_components`, `nocv_eigenvalues` (the +nu_k of each pair, signed tail), `orbital_components_alpha`/`beta`, `nocv_eigenvalues_alpha`/`beta` | hartree; eigenvalues electrons |
| `density_at_nuclei` | `values` — rho(R_A) per atom in the molecule's canonical atom order | electrons/bohr³ |
| `qtaim` | `bond_critical_points` (each: `atom_a`, `atom_b`, `position_bohr`, `density`, `laplacian`, `ellipticity`, `eigenvalues` ascending, `bond_path` — the two gradient rays BCP -> nucleus), `other_critical_points` (`position_bohr`, `rank`, `signature_sum`), `unconverged_seeds` (3 each) | electrons/bohr³; electrons/bohr⁵ |
| `molden` | `file` — the path of the written Molden-format F-file | path |

### 3.5 Null vs absent — stated precisely

A JSON consumer must distinguish three cases:

1. **Key present with a value** — computed, real output.
2. **Key present with null** — applies to `spin_squared` (RHF: not
   applicable), `populations`/`moments` (block never computed), and the
   inner `charges.hirshfeld`/`charges.voronoi` members (that analysis not
   requested). The key exists; the value is null.
3. **Key absent** — applies to the opt-in analysis blocks (`charges`, `esp`,
   `eddb`, `fukui`, `nalewajski`, `nocv`, `density_at_nuclei`, `qtaim`,
   `molden`) when the run did not request the analysis, to
   `certified_bound` (§3.9) when no Fock build reported the quantity, and to
   `qfmm_model` (§3.10) when the run wired no composed-QFMM builder.
   The key does not exist at all.

The serializer never emits a placeholder number for an unset value — null
(or absence) means "not computed", never "zero".

### 3.6 `resources_resolved`

The caps the run was actually executed under — the `[resources]` block
resolved (the resource caps).
Always present, defaults included, so a JSON consumer can always see the
enforcement state of a run.

| Key | Type | Meaning |
|---|---|---|
| `memory_cap_gib` | number | The process-memory ceiling in GiB the run was executed under (16.0 default; 0 = no input cap) |
| `thread_cap` | int | The OpenMP team ceiling the run was executed under (0 = all detected hardware threads; 1 = the serial path) |
| `in_process_cap_applied` | bool | Whether the driver's own hard cap was actually applied (false when already inside an external job — the benchmark harness gate or a CI runner — or on platforms without Windows job objects) |
| `cap_note` | string, present only when the cap was not applied | Why the hard cap was not applied (an audit note, never a failure) |
| `workspace_budget` | object, present on the budget path | The cap-minus-base workspace budget granted to the engine (the admission arithmetic): the capacity and the cumulative Create-time commit — ri_j_link, the direct family's adaptive seam, and the composed-QFMM runs (one shared budget reaches both nested halves, so the commit covers both Creates). Absent on the legacy null-budget path (cap 0, or a cap at/below the modeled base) and for the gpu family |
| `mode_record` | object, present on the budget path | The engine's Create-time rung decision with its firing estimate terms — the budget path's executed-peak evidence (mode, predicted bytes, exclusions). On the composed-QFMM runs the record is the Coulomb (QFMM) half's decision; the exchange half rides the same shared budget (its commit is inside `workspace_budget.committed_bytes`, and the UHF run surfaces it under `exchange_mode_record`) |
| `exchange_mode_record` | object, present on UHF runs that reached the budget path | The UHF exchange half's rung decision (the UHF Fock builds from TWO builders — the direct coulomb/exchange pair or the composed-QFMM halves — and both admission-gate observations must be visible) — same shape as `mode_record`; absent on RHF runs (single-builder direct/ri_j_link; the composed-QFMM RHF half is not surfaced separately) and on the legacy null-budget path |
| `selection` | object, present on every run that reached the resolution (the last member of the block) | The builder resolution record (the selection shape, under the lean-direct flip): what the run wires without an explicit key, why, and how an explicit request diverged from it, if at all. No model steers it any more — the pick is the direct-family default at every size |
| `ri_tensor_mode` | object, present whenever the input NAMED either request key (`method.ri_tensor_mode` or `[diagnostics] force_disk_ri`) | Schema 19; schema 22 adds the `forced` member and the force request. The requested-vs-ran pairing of the RI tensor mode: `requested` / `resolved` / `outcome` / `reason` / `forced` (the block paragraph below, with the four outcome words). It exists because the mode record alone cannot answer what the input asked for on the paths where a request would otherwise leave no trace — a run with no `mode_record` (the lean arm, the legacy null-budget path) and every family with no RI disk rung (a `disk` request on a direct/qfmm/gpu/lean run). Absent when both keys were omitted: an omitted key is the default request, and the disclosure rule makes a default the system's judgement rather than a request to record |
| `compute_profile` | object, present on every run that resolved a compute profile | Schema 20; schema 21 adds `certified_lane_forced`. The hardware input the certified fp32 lane's default was RESOLVED against, with the verdict read off it and the threshold it was compared against — the lane's *decided* record, where `certified_bound` is the same lane's *delivered* one (the block paragraph below). Unlike `ri_tensor_mode` this records no request — with ONE exception, and it is the exception that keeps the block readable: `certified_lane_forced` says the run's lane state did NOT come from this machine's verdict, because the input forced it (`[method] force_certified_lane`, the owner's ruling 2026-09-13). A caller's explicit `use_certified_mixed_precision` request still resolves at one point and is not a property of the machine; the flag is the statement that it was made |
| `ri_chunk_bytes` | object, present whenever the input NAMED `method.ri_chunk_bytes` | Schema 23. The `ri_chunk_bytes` request record: `requested_bytes` / `outcome` / `reason` (the block paragraph below, with the two outcome words). It exists for the same reason its sibling does, on the key that had no record at all: the size hint is read at exactly ONE site — the `ri_j_link` family's disk-rung construction, where it lands in `DiskRiFockOptions.chunkBytes` — so on every other family, and on `ri_j_link` when a fitting in-memory rung rides, the run proceeded and the key was dropped without a word anywhere in the document. The drop itself is the owner's ruled behaviour (2026-09-13: refusing a size hint on a family with no chunking to size is pedantic); the disclosure is this block. Absent when the key was omitted: an omitted key is not a request |
| `ri_orbit_expansion` | object, present whenever the input NAMED `method.ri_orbit_expansion` | Schema 26. The orbit-expansion request record: `requested` / `outcome` / `reason` (four outcome words below). It exists for the same reason its siblings do, on a key whose consumption is one branch deep: the engine it drives is inert until the `ri_j_link` arm hands it a reduction, so without this block a run that asked and engaged, a run that asked on a trivial group and engaged nothing, and a run that asked on another family altogether would serialize identically. `requested` is always written (a false request is a fact about the input); `reason` is absent when the outcome is `engaged` (the null honesty policy). Absent when the key was omitted: an omitted key is not a request |

**Schema 36 REMOVED the `memory_model` object** (its keys were `builder`,
`n_basis`, `n_aux`, `tensor_gib`, `copies_factor`, `base_gib`,
`modeled_peak_gib`, `cache_budget_gib` and `lean`), and every key under it,
together with `driver/src/memory_model.cpp` and its header. It was the
driver's hand-derived prediction of a run's commit peak, and the owner's
ruling of 2026-09-17 deleted it rather than repaired it: a hand-derived
formula drifts from the allocator by construction, and it was caught wrong
by 5×, 15× and 81 percent across its terms plus an extrapolation from a
fitted span to nine times beyond it. What still holds a run's memory is the
job-object cap (`memory_cap_gib`, `in_process_cap_applied`, `cap_note`
above — the enforcement path is untouched by the removal), and what still
decides the rung a run takes is the integrals engine's own Create-time
estimates, reported under `mode_record`.

The one fact the removed block carried that no other key carried is the
direct family's within-family member. It is not lost: the selection
record's `builder_member` (schema 15) names the member itself rather than
flagging it, so a consumer reads the choice off that key.

The `workspace_budget` object — bytes, both monotone
cumulative counters of the engine's Create-time reservation:

| Key | Type | Meaning |
|---|---|---|
| `capacity_bytes` | int | The cap-derived capacity granted: `memory_cap_gib`·2³⁰, floored at 0 (schema 36: the driver's modeled base term is gone, so the whole cap is the budget and the engine's own Create-time estimate decides the rung against it) |
| `committed_bytes` | int | The engine's cumulative Create-time commit against the capacity (the admission arithmetic's final state) |

The `mode_record` object — the engine's rung decision and its firing estimate
terms (absent on the legacy null-budget path, where no decision was made).
`mode` is the rung that fired; the byte keys are the admission arithmetic's
firing terms, `predicted_bytes` the firing estimate. On the driver's disk
route the record is DRIVER-SYNTHESIZED — no engine record exists there (the
ladder refused before deciding, and integrals cannot link storage): `mode` is
`"kDisk"`, `disk_bytes` the modeled dense on-disk payload,
`budget_bytes`/`remaining_at_decision` the budget context the refusal left
untouched, all other terms zero. Two admissions reach that route and the
record names which one did: `forced_disk` (schema 18) is `true` only on the
forced route (`[diagnostics] force_disk_ri = true`, schema 22 — the key that
replaced the `ri_tensor_mode = "forced_disk"` word, where no ladder ran at
all) and `false` on the ladder's own fallback, so a measurement cell can never
be read as a run the ladder sent to disk:

| Key | Type | Meaning |
|---|---|---|
| `mode` | string | `"kFastPath"` \| `"kLightPath"` \| `"kDisk"` — the rung that fired (the engine's enumerator vocabulary; `kDisk` appears only on the driver's disk route — no engine sets it). The two admissions of that route are told apart by `forced_disk`, not by a second mode word: the store that ran is the same one, and a consumer that filters disk cells keeps working |
| `forced_disk` | bool | Schema 18. Whether this record came from the FORCED diagnostic route (`[diagnostics] force_disk_ri = true`; the retired `method.ri_tensor_mode = "forced_disk"` spelled the same route before schema 22) rather than from a decision: `true` on the forced route — where the driver skipped the composed in-memory ladder entirely, so no `RiJkFockBuilder::Create` ran and `predicted_bytes`/the engine terms carry no decision (zero, as on the ladder's disk fallback) — and `false` on every engine rung decision and on the ladder's own disk fallback. Always present on a serialized record: an absent member could not distinguish the two admissions |
| `predicted_bytes` | int | The full footprint estimate of the admitted rung — the estimate the mode decision was made against, not the executed peak (the first-iteration tensor-class allocations are unmodeled) |
| `reserved_bytes` | int | Bytes actually reserved |
| `budget_bytes` | int | The workspace capacity the decision saw |
| `remaining_at_decision` | int | Capacity − committed at the decision point |
| `max_batch_bytes` | int | The largest shell-pair batch |
| `pair_store_bytes` | int | The pair store |
| `pattern_bytes` | int | The pattern-space table |
| `scratch_bytes` | int | The engine scratch |
| `structural_bytes` | int | The shell/block structures |
| `cache_bytes` | int | The ERI cache (0 when off) |
| `light_store_bytes` | int | The light rung's pair store |
| `chunk_arena_bytes` | int | The light rung's chunk arena |
| `chunk_pattern_bytes` | int | The light rung's pattern table |
| `chunk_index_bytes` | int | The light rung's chunk index |
| `light_shells_bytes` | int | The light rung's shell table |
| `chunk_pairs` | int | The light rung's chunk pair count |
| `disk_bytes` | int | The disk rung's modeled dense on-disk (uv\|P) payload `8·n²·nAux` (chunk stores are dense, so this is also the store file's payload). Zero on in-memory records; named in the model but never reserved from RAM (the driver's disk route synthesizes the record) |
| `tensor_bytes` | int | The dense ri_j tensor |
| `ri_matrix_bytes` | int | The retained n^2 x nAux Eigen copy of the tensor (the Create-time riMatrix); the metric factorization is `metric_bytes` |
| `task_list_bytes` | int | The SCF task list |
| `metric_bytes` | int | The metric factorization |
| `orbital_aux_bytes` | int | The aux-orbital intermediates |
| `outer_store_bytes` | int | The outer-shell store |
| `exchange_bytes` | int | The direct exchange |
| `exchange_scratch_bytes` | int | The exchange scratch |
| `pattern_excluded` | bool | Whether the pattern term was excluded from the estimate (the exclusion rules) |
| `tensor_excluded` | bool | Whether the tensor term was excluded (the C24H50/C60 light-path evidence: tensor excluded → the light rung) |
| `class_table_bytes` | int | The class-path table's footprint at Create: the on-demand restructured class table the admission gate sized. The sure-fit evidence — the class path disengages (see `class_path_disengaged`) rather than let this table overrun the budget |
| `class_path_disengaged` | bool | Whether the class path disengaged at Create: the symmetry-reduced class table could not fit the budget, so the plain screened path runs — never a hard failure, never a silent one (this flag is the record) |
| `concurrent_slots` | int | Schema 12: the Create-time authorized concurrent batch slots (the bounded-concurrency batch loop's k); 1 when the slots were never authorized; 0 only on the driver-synthesized disk record (no engine decision produced it). The runtime per-pass kEff can sit below it — the *observed* per-call concurrency is a FockBuildStats read (`FockBuildStats::concurrentSlots`), not part of this record. Schemas 7–11 never carried this field (the k a fired run carried was lost in the JSON; an early diagnosis read it from the process log) |
| `default_team_size` | int | Schema 12: the decision's clamp-origin team read — `DefaultOmpTeamSize()` at Create on the budgeted path, the team the `k = min(...)` decision clamped against. A record with `concurrent_slots == default_team_size` ran team-clamped (k = the whole ceiling-bounded team, the k=5 reading of that diagnosis); one below it ran budget-clamped. 0 on the legacy no-budget path (no decision, no read) |

The `exchange_mode_record` object — the UHF exchange half's rung decision.
UHF Fock builds from TWO builders — the direct coulomb/exchange pair or
the composed-QFMM halves — and both admission-gate observations
must be visible, so the exchange half's record serializes under its own
key with the same object shape as `mode_record`. Absent on the RHF runs
(single-builder direct/ri_j_link; the composed-QFMM RHF run's exchange
half is not surfaced separately) and on the legacy null-budget path.

The `selection` object (absent members are omitted — the null honesty
policy of §3.5). Its shape is the selection record's, but the lean-direct flip
(2026-09-08) retired the model-driven pick from the default path: the
no-builder default is the DIRECT family at every size (the lean
Schwarz-only builder at n ≤ 1000, the budgeted direct machinery above),
and `ri_j_link` / `qfmm` / `gpu` run only as explicit opt-ins. One choice
spans two keys: `builder` names the FAMILY, `builder_member` (schema 15)
the member of it that actually ran:

| Key | Type | Meaning |
|---|---|---|
| `builder` | string | The builder the run actually wired — the FAMILY word |
| `builder_member` | string, present on every run that reached the resolution | Schema 15. The within-family MEMBER the run wired — the key `builder` cannot carry, since the family word stays the family word. `"lean"` when the direct family wired its lean (Schwarz-only) member (the auto-lean default at nBasis ≤ 1000, or the explicit `fock_builder = "lean"`); the family word itself on every other run (the machinery — an explicit `fock_builder = "direct"`, its alias `in_memory`, or the absent key above the ceiling — and the RI-J/QFMM/GPU families). Always one of the two, so a consumer READS the member; `explicit_builder`'s absence is not that signal (the lean carve-out leaves it absent by design). Pinned against the family word by the driver's consistency test |
| `picked` | string, present on every run that reached the resolution | The builder the run wires WITHOUT an explicit `fock_builder` — the direct-family default at every size under the lean-direct flip, never a model pick (the ranking is retired). The driver-synthesized disk record carries no resolution and sets none |
| `reasoning` | string | The resolution's why, verbatim, one line. The opener names the request: `no fock_builder given: ...` for an omitted key, `explicit fock_builder <word> honored ...` for a request — the lean request carries its own text (it names no family word, so it must not borrow the omitted-key opener) |
| `explicit_builder` | string, present when the input named a builder family word | The input's `fock_builder` value (the request that won or fell back), in its CANONICAL spelling: `in_memory` is an accepted alias of `direct` (§2.1, the tier axis), both resolve to `kDirect`, and `kDirect` is spelled `direct` — so this member names the family that owns the requested tier, and an author who wrote the tier word reads back the family word for the same tier (`ToString`'s note, `io/include/qcx/io/run_input.hpp`). The lean carve-out: `fock_builder = "lean"` is the direct family's within-family member, not a family word, so it leaves this member ABSENT by design — the request is carried by `reasoning` and the choice it makes by `builder_member` |
| `warning` | string, present when the run diverged from the pick | The divergence text. Under the lean-direct flip exactly one divergence is left — the device-less `gpu` fallback ("no CUDA device is present"): an explicit request never diverges from the pick any more (the model that ranked it is retired), and the UHF direct restriction died with it. The driver also prints it to stderr as `qcx: warning: ...` — never a silent substitution |
| `approximation` | object, present when the run wired an approximate exchange builder **or** the aux selection carries a notice | Schema 24; schema 25 adds `aux_notice` and widens this presence rule. The exchange-approximation disclosure: a run whose exchange half is not the exact two-electron kernel must say so, or a consumer reads an energy built from an auxiliary fit as though it came from exact quartets. Members: `exchange` — the contraction form the exchange was built in (the member word, so this and `builder_member` name the same thing), reading `"exact"` on a notice-only block; `aux_basis` — the auxiliary basis name IN EFFECT (the explicit `[basis].aux` when the input carried one, else the resolved auto-selection, so an auto-selected default never reads back as a user request); and `aux_notice` — the auxiliary-basis SELECTION notice, why the aux was chosen by a policy that warns for the region it is in (minimal bases, bases without polarization, diffuse-augmented bases, ECP/relativistic bases, and a JK request with no matched JK fit). **`aux_notice` is absent rather than null** when the selection is in no weak region, the same null honesty the block's own absence uses one level up (cf. §3.5). Schema 25 **widens the presence rule**: the block is now emitted when a notice exists even on a run whose kernels are exact, because a notice hanging off an absent block is invisible and a reader must not have to infer that the block was omitted for the exact-kernel reason. A consumer keyed on the old "absent on every exact run" rule is the one that will notice. **The block's `aux_basis` is the error class, and `[method] accuracy` does not bound it:** on an approximated-exchange run the preset governs the SCREENING CO-TERM ONLY, so a consumer must read the run's accuracy from the auxiliary fit this block names (whose error against exact K is measured per atom by the exchange-accuracy cell) and never from the preset. **Schema 31 adds `exchange_error`** — the measured-error disclosure, and the operative half of the owner's 2026-09-16 ruling that the approximated-exchange path closes with its error DISCLOSED. Its members: `measured_per_atom_hartree` — the achieved per-atom exchange error of the path, measured against exact screened exchange at the same preset; `measured_on` and `measured_aux_basis` — the fixture and the auxiliary fit that measurement was taken on, i.e. the SCOPE of the number beside them; `bar_per_atom_hartree` — the bar THIS run is read against (`RiExchangeErrorBudgetPerAtom`, the preset's own per-atom budget); and `bar_preset` — the preset word that bar belongs to, emitted because the record echoes `[method] accuracy` nowhere else and a bar a reader cannot tie to a preset is unattributable. **Present exactly when `exchange` names an APPROXIMATED contraction — a NARROWER rule than the block's own**, which schema 25 keeps on exact-kernel runs carrying a notice: on that arm the fitted exchange's error is no error of that run's, so writing it would be a lie about the run rather than a disclosure of it. **The two numbers are of different kinds, and the record never presents them as one**: the bar is the run's OWN, while the measured error is the PATH's — no run can compute its own error (evaluating exact K is the cost the path exists to avoid, `ri_full_fock.hpp`), so what ships is the worst of the characterized cells with its fixture named. It is a measurement, **NOT a bound**: a run on an uncharacterized fixture may sit on either side of it. The values and their ratios per fixture x preset are the accuracy cell's to state (`integrals/tests/ri_jk_validation_test.cpp`), never restated here; the mapping a consumer compares the key against is `RiExchangeWorstMeasuredPerAtomError` (see §5's `ri_jk` bullet) **Schema 32 adds the top-level `symmetry_blocking` block** with its two channel members `alpha` and `beta` (`RunSymmetryBlocking`), the blocking guard's own decision record: `action` — the guard enum word, one of `kUsed` (every diagonalization ran on the irrep blocks), `kDemoted` (some ran blocked and some ran the plain solve, i.e. the run walked BOTH paths), `kRefused` (blocking was requested and never ran) or `kUnavailable` (the linear-dependence removal left a non-square orthogonalizer, so no blocked solve could be served and no Fock was measured) — beside `blocked_solve_count`, `plain_solve_count`, `generator_commutator_norms` (the relative commutator norms of the WORST measurement of the channel, one per generator of the computational group, so a reader can VERIFY the decision instead of taking it), `max_generator_commutator_norm` and `tolerance` (the threshold the norms were compared against, emitted so they are readable without the source). It exists because the blocked diagonalization equals the plain one only while the spin Fock commutes with the group — a property of the SOLUTION, not of the nuclear framework — and a blocking that was demoted mid-run used to be visible only to a caller holding the in-process result, so the run's own JSON read as though the irrep blocks had been used. **Presence rule:** the key is ABSENT when nothing was requested (no basis set supplied, a C1 molecule, an unrealizable group — the guard enum `kNotRequested`, which the absence states rather than a fabricated word), and inside a present block a channel that disclosed nothing is null, never a fabricated `kUsed`; a consumer therefore cannot read blocking-was-used off a run that never asked. The value is the action the DECIDING code produced (`scf/src/uhf.cpp`'s guard, through the driver's `MakeRunSymmetryBlocking`), never re-derived from the counts beside it. **Schema 33 adds `resources_resolved.eri_store`** (`RunEriStore`) — the disk-tier ERI store request's requested-vs-ran pairing, the disclosure half of the engine-decorator seam (`method.eri_cache_store`). Members: `path` — the store path the input named, VERBATIM, and written on the demoted arm too (a disclosure that did not say what was requested could not be read as one); `engaged` — what actually RAN, one of `"disk"` (the storage-module decorator served the run's ERI batches, the request honoured), `"in_memory_cache"` (the run proceeded on a tier that is not the store) or `"none"` (no cache tier served it at all); `demoted` — the flag saying the requested store was NOT what ran, so a consumer never has to infer the demotion from the `engaged` word's value, and `demoted_reason` — the cause, VERBATIM from the store's own error (a refused open, a write failure, a fingerprint mismatch), absent on an honoured request (the null honesty policy); and the decorator's own counters, `hit_quartets` / `miss_quartets` / `read_ms` / `recompute_ms` mirrored from `CachedEriStats` after the SCF with `hit_quartets_by_class`, an ARRAY of `{l_bra, l_ket, hit_quartets}` rows so the class a count belongs to is written beside it rather than encoded in a composite key. The last four are 0 when the store did not engage — never a fabricated reading, and never a substitute for `engaged`. It exists because the seam by itself was unreachable from any input: the decorator is chosen OUTSIDE the builder (`integrals` cannot name `storage`, so the driver constructs it, owns it and reads its stats), which leaves the run's own JSON the only place a consumer can learn whether a disk request was honoured. **Presence rule:** the block is present whenever the input NAMED a store (`method.eri_cache_store` non-empty), on BOTH the honoured and the demoted arm — it is the demotion's disclosure surface and not only the success path's evidence — and ABSENT when the key was omitted or empty, because an omitted key is not a request (the schema-19 null-honesty rule its siblings follow). `hit_quartets_by_class` is omitted rather than written as an empty array, the block's own absence rule applied one level down. An enforced request is therefore USED (`engaged` = `"disk"`, `demoted` = false) or DEMOTED WITH THE DEMOTION DISCLOSED (`demoted` = true beside the untouched `path`); a request REFUSED BY NAME is a run that never reached serialization, so it has no shape here — and a silent substitution, the in-memory tier running under a disk request with nothing in the record to say so, is the defect this block makes unreportable. **Both halves are WIRED** (the driver call site landed 2026-09-17): the input key and the record block land together here, and `driver/src/run_driver.cpp` reads `method.eri_cache_store`, installs the decorator (`FockBuildOptions::engineDecorator`, called once at the builder's own Create) on the direct family's machinery member, owns the store for the run and fills this block from the decorator's stats after the SCF. Three things a consumer must know about what that site does with the request. **(1) The tier is read from the store's OWN traffic, never from the install.** The builder disengages the whole engine tier - the decorator with it - on the class-aware path (a nontrivial symmetry reduction) and on the LightPath, and both decisions are taken INSIDE Create, after the driver has handed its factory over: on those runs the factory is never called, no store is ever built, and the block reports a demotion that names the mechanism instead of claiming a disk tier that served nothing (measured: an H2/STO-3G direct run at the default cap reports `demoted` = true with `class_table_bytes` > 0 in the same document). **(2) The demotion's causes are three different sentences** - the store's own error VERBATIM (an unwritable path, an HDF5 failure, a fingerprint mismatch), the family that carries no batch engine pair to decorate (the lean member, ri_j_link, qfmm, gpu), and the class-aware disengagement above - and in every case the run COMPLETES on the closest workable arrangement with its arithmetic unmoved, because the decorator serves the engine's own bytes. **(3) The unrestricted legs REFUSE the key by name** (verbatim: *not yet wired on the UHF path*): the per-spin wiring assembles its own options struct, so an explicit request there is answered rather than ignored - and since a refusal is a run that never reaches serialization, a refused run has no block here. **`hit_quartets_by_class` is filled from the store's own per-class counters, which the store records for single-class requests only** (its documented scope - `storage/src/cached_eri_batch_engine.cpp`, the benchmark's request shape): an SCF run's requests span classes, so an ordinary run's block OMITS the array while its `hit_quartets` is non-zero, and a consumer must never read the array's absence as an absence of hits. **Schema 34 adds the run's own PHYSICS, under ONE bump for two payloads** — the top-level `method` and `functional`, and the top-level `xc_grid` block. `method` (always present) is the word the run's dispatch acted on, one of `"rhf" \| "uhf" \| "rks" \| "uks"` — the input's own `[method] type` vocabulary, spelled through the io function that is ParseMethod's inverse, so a consumer reads in the artifact the word it can write back into a file; it is the path the driver's ONE classifier resolved rather than the input enumerator re-read, so the word and the loop that produced the numbers cannot disagree. `functional` (present on a Kohn-Sham run, **absent** on every Hartree-Fock run) is the RESOLVED registry name the input's `[method] functional` resolved to, never the file's string re-read. `xc_grid` (present on the same runs, absent on the same runs) is the six grid settings the run BUILT (`RunXcGrid`) — `radial_points`, `angular_points`, `alpha`, `radial_exponent`, `trim_weight`, `block_target` — the resolved `[grid]` block the engine's ONE `Create` was called with, so the record names the grid it used and never merely the one it was asked for; like `functional` it is ABSENT rather than null or zero-filled where no density functional was named and no XC grid was built, because a block of defaults would be a claim about a quadrature that never touched those numbers (the null-honesty rule the `symmetry_blocking` absence uses one block over), and a `[grid]` block written on such a run is REFUSED by name (§2.3) rather than ignored — the absence in the record is then always the honest kind, since the only way to reach it with grid keys in the file is a run that never reached the record at all. It exists because the record named its Fock builder and no physics at all: an `rks` run and its `rhf` twin were distinguishable only by their numbers, and the grid — a physics choice, since the same input at a different Lebedev size is a different energy — was compile-time settings no input could reach and no record could name. No existing key moved, was removed, or changed meaning. |
| `file_version` | int | Schema 11: the loaded coefficients file's `file_version` (the provenance of every score below) |
| `machine_class_key` | string | Schema 11: the run's machine-class key — the fit dimension the measured cells were grouped by |
| `basis_family_id` | string | Schema 11: the run's basis-family id (the fit's other dimension) |
| `preset_id` | string | Schema 11: the run's method preset ("default" in v1) |
| `cell_key` | string, present when the best selectable candidate's score came from a measured cell | Schema 11: the matched cell's canonical key |
| `borrowed` | bool | Schema 11: the best selectable candidate's score came from a borrowed cell (the synthetic new-family placeholder, donor times `inflation`) |
| `inflation` | number, present exactly when `borrowed` is true | Schema 11: the inflation applied to the donor score |
| `scores` | array of objects | Schema 11, RETIRED PRODUCER: the shape is `{ "candidate": "direct", "score_seconds": 1.234, "source": "analytic" \| "table" \| "borrowed", "selectable": true }` and the serializer still renders it, but **nothing in a real run populates it** — the candidate-ranking heuristic it reported was retired by the lean-direct flip, and the field's only remaining producer is its own serializer test (`io/tests/result_json_test.cpp`). Whenever the block is emitted the array is EMPTY — the render loop has nothing to iterate — and a run with no coefficient file omits that block, so no `scores` member appears at all. It stays in the schema because a coefficients record's shape is versioned (`file_version`) and dropping a documented member is a version bump, not an edit — but a consumer must not read it as evidence that a ranking happened. The decision it used to carry is in `picked` / `builder_member` / `reasoning` |

The `ri_tensor_mode` object (schema 19; `forced` added in schema 22) — the
requested-vs-ran pairing of the RI tensor mode. It is written whenever the
input named EITHER request key — `[method] ri_tensor_mode` or
`[diagnostics] force_disk_ri` — whatever ran and whichever path the run took,
because the enforcement
contract's question ("did the request get what it asked for?") cannot be
answered by the mode record on the runs where a request would otherwise leave
no trace at all: the lean arm and the legacy null-budget path carry no
`mode_record`, and a family with no RI disk rung never had one for this
question. `requested` is the request's word — `auto` or `disk` for the rung
selector, `forced_disk` for the force, which keeps the word because it names
the REQUEST and no longer exists as a rung word; `resolved` is `"disk"`
when the storage-module disk-backed builder was wired and `"in_memory"` on
every other path (the engine's own rung is `mode_record.mode` where that
record exists); `outcome` relates the two, and `reason` explains it — present
unless the outcome is `"honoured"` (an absent reason is never a fabricated
"fine"). `forced` (schema 22) names the KEY the request came from: `true`
exactly when `[diagnostics] force_disk_ri` carried it, `false` for a rung
word — always present, so a reader can tell the two request sources apart
without the input file (a demoted forced request still reads `forced: true`;
`outcome` states what happened to it). The four outcome words:

| `outcome` | Meaning |
|---|---|
| `honoured` | The request got what it named: `auto` → in-memory, and `disk` / `forced_disk` → disk. The REQUEST did not have to change |
| `ladder_fit` | `disk` was permitted and a fitting in-memory rung rode. **Not a demotion**: the knob's documented inert-by-design contract — `disk` permits the ladder's last rung, it does not demand it — so a run that stays in memory is exactly what the request prescribes. Reporting it as a fallback would be the record lying in the direction opposite to the one this block exists to fix |
| `not_applicable` | The request had nothing to select on this run: the RI disk rung is the `ri_j_link` family's route, so a `disk` request on a direct / qfmm / gpu / lean run is recorded rather than silently dropped (the row the disclosure audit found missing) |
| `demoted` | The request could not be honoured and the best runnable arrangement ran instead, stated as such (a disclosed demotion is the expected outcome; a silent substitution is the forbidden one). `forced_disk` (`[diagnostics] force_disk_ri = true`, schema 22) on a family with no `ri_j_link` disk route is the case in v1 |

The `ri_chunk_bytes` object (schema 23) — the outcome of the disk rung's
chunk-size hint, and the disclosure half of the key-split rule. It is written
whenever the input named `method.ri_chunk_bytes`, whatever the family and
whatever the run did with it: the hint's only consumer is the `ri_j_link`
family's disk-rung construction, so on a direct / qfmm / gpu / lean run — and
on `ri_j_link` when a fitting in-memory rung rides — the run proceeds exactly
as it would have without the key, and this block is the only place the
document says so. `requested_bytes` is what the input asked for, in bytes,
recorded in BOTH outcomes (a disclosure that did not say *what* was dropped
could not be told apart from any other key's drop); `outcome` is `"honoured"`
(the storage-module disk-backed store was constructed and took this chunk
size) or `"dropped"` (the run proceeded without it); `reason` explains a drop
— present on `"dropped"`, absent on `"honoured"` (an absent reason is never a
fabricated "fine").

The block deliberately does **not** reuse the sibling's four outcome words.
`ladder_fit` on `ri_tensor_mode` means the request was NOT lost — a fitting
memory rung is what a `disk` request prescribes — while the same physical run
DROPS this hint, because the rung that would read it never engaged. One word
with opposite polarity in two sibling blocks would have to be decoded rather
than read, and the two blocks are read side by side: one run emits
`ri_tensor_mode.outcome = "ladder_fit"` and `ri_chunk_bytes.outcome =
"dropped"`, and each statement is true of its own key. The two drop texts
follow the sibling's discipline (two facts, two texts): a family with no disk
rung at all, and the `ri_j_link` route that has one but did not engage.

The `compute_profile` object (schema 20; `certified_lane_forced` added in
schema 21) — the machine's measured fp32/fp64
FMA-throughput ratio, and the verdict the certified fp32 lane's default was
read off it. The quantity is a RATIO and never a label: the certified lane's
enablement asks whether fp32's throughput premium covers the lane's routing
and conversion overhead, and a label answers a different question — this tree
has already had a wrong label in exactly this place (the FMA flag read from
the wrong CPUID leaf, which made the whole AVX2 micro-GEMM tier dead code on
a machine that has FMA). A measured ratio cannot be wrong that way; it can
only be noisy, which is why its noise rides along.

| Key | Type | Meaning |
|---|---|---|
| `source` | string | Which probe produced the numbers: `"host"` (the CPU-side register-bound SIMD FMA micro-benchmark, `backend/host_compute_profile.hpp`) or `"device"` (the CUDA probe, read behind its `GpuProbeFloorGiB` cap gate). The driver selects it from the family the run WIRES — the host arm for a CPU family, the device arm for a GPU one — so a device number can never arrive as a host reading or the reverse. That inversion is not hypothetical: a device ratio of 31.2 reaching a CPU builder would turn the *host* lane ON on a machine that measured 2 |
| `fp32_gflops` / `fp64_gflops` | number | The measured per-precision FMA throughput, scalar FLOP and SIMD lane width included (0.0 = unmeasured). Clock- and core-dependent: never compare them across machines. The ratio below is the portable quantity |
| `ratio` | number | `fp32_gflops / fp64_gflops` (1.0 = unmeasured). On a mainstream x86 host the peak is a SIMD lane-width fact — 8 fp32 lanes against 4 fp64 lanes under AVX2 — so it reads ~2, half the lane's 4.0 threshold; a scalar host reads ~1 |
| `measured` | bool | Whether an actual measurement produced the numbers above. This is what keeps a MEASURED 1.0 (a scalar machine, no fp32 premium) apart from the UNMEASURED 1.0 (no probe at all). The verdict is off either way — the record is not |
| `simd_lane` | bool | Whether the host probe ran its SIMD lane (256-bit AVX2 FMA) or its scalar reference lane. Always false for a `"device"` source |
| `pairs` / `ratio_spread` | int / number | The timed lane pairs the ratio was estimated from, and the spread of the per-pair ratios (max − min). The estimator is a min-of-pairs peak; the spread is the instrument's own noise, recorded so a reading can be judged rather than trusted. Both are 0 for a `"device"` source, whose probe measures once |
| `certified_lane_default` | bool | The verdict: the lane's default for this machine, i.e. `ratio > certified_lane_min_ratio`. The lane's DEFAULT, never a caller's explicit request — and since schema 21 never the lane's state on a forced run either: read `certified_lane_forced` beside it |
| `certified_lane_forced` | bool | Schema 21 (the owner's ruling 2026-09-13). `true` when the input FORCED the lane through `[method] force_certified_lane` rather than letting the verdict above decide it — the one member of this block that is not a property of the machine. **Always present**, on the `mode_record.forced_disk` rule: an absent member could not distinguish a forced run from a probed one. The block keeps publishing what the probe measured and what its verdict would have been, so a forced run reads `certified_lane_forced: true` BESIDE its own `certified_lane_default` — never infer the lane's state from `certified_lane_default` alone |
| `certified_lane_min_ratio` | number | The threshold the ratio was compared against — `qcx::integrals::kCertifiedLaneMinRatio` (4.0), carried rather than copied so a consumer does not have to re-derive the rule. io cannot include the integrals policy; the driver fills it from the one definition |

The block is present on every run that resolved a profile and absent when none
applied. A device-arm run whose cap gate closed still carries the block with
`source: "device"` and `measured: false` — the arm applied and measured
nothing, which is a different statement from "no probe applies".

### 3.7 `symmetry` and `symmetry_beta`

The post-SCF full-group labeling record (`[symmetry] full_group` is
on by default). `symmetry` carries the RHF channel or the UHF alpha
channel; `symmetry_beta` the UHF beta channel (UHF runs only). Both are
**absent** when the stage did not run — a C1 molecule (the trivial group
carries no information), `[symmetry] full_group = false`, a group the
stage cannot realize, or **the linear-dependence removal** (§3.1) — never a
fabricated C1 record. The fourth cause is the one a reader cannot infer from
this block alone, and it is stated in the form the record is read in:

> **Point-group blocking was requested for this run and DEMOTED.** The
> linear-dependence removal left the working space `numKept`-dimensional, so
> the blocked diagonalization fell back to the full `n`-dimensional solve and
> the full-group labeling/symmetrization stage did not run. **Reason:** the
> removal is a numerical threshold on the overlap spectrum and does not
> respect the group action, so the kept space is a union of irreps only by
> accident — the stage's contract (one label per MO column of the
> `n`-dimensional set, the full-group average of the `n x n` density) is a
> full-space contract. **Evidence:** this run's own
> `num_removed_overlap_directions`, nonzero exactly when the demotion (the guard's own record names this case `kUnavailable`, and `kDemoted` there means a run that walked BOTH paths — schema 32, `symmetry_blocking`)
> happened; a run that removed nothing is UNUSED-AND-UNCHANGED, and its count
> is 0.

That is the whole record of the demotion: the count is the fact and this block
is its consequence, so nothing is stated twice. The MO coefficients
are not repeated here: the Molden export (§3.4) carries them; the labels
and the subspace records reference MO columns in coefficients order.

| Key | Type | Meaning |
|---|---|---|
| `full_group` | string | The detected full point group (`"C2v"`, `"Dinfh"`, ...) |
| `abelian_reduction` | string | The computational (Abelian) group the SCF actually used — the reduction of the full group |
| `labels` | array of strings | The per-MO Mulliken labels, one per coefficient column, in coefficients order |
| `irrep_indices` | array of ints | The irrep index of each MO (the Abelian-group column of the labels) |
| `canonicalized` | array of objects | The degenerate subspaces the stage canonicalized (rotated to diagonalize the Fock sub-block): `{ "irrep_label": "E", "mo_indices": [..] }` |
| `straddled` | array of objects | The degenerate subspaces that straddle a canonicalization boundary and kept the SCF's arbitrary partners: `{ "irrep_label": "E", "mo_indices": [..] }` |
| `symmetrization_subset` | array of strings | The finite irrep subset the density symmetrization averaged over |
| `averaged_element_count` | int | The number of elements averaged by the symmetrization |

### 3.8 `term_counters`

The cost-calibration block of an instrumented `ri_j_link` run:
the engine-emitted subset of the cost-table backbone's `ri_j` term vector
(the record contract's `EMITTED_COUNTER_VECTORS["ri_j"]`). Exactly five
keys, all non-negative INTEGER per-run totals over the whole SCF run (not
per iteration); a term whose work never occurred reports 0 — never
omission, never null:

```json
"term_counters": { "x": 1726, "p3": 118820, "g3": 16682642, "qx": 4017936, "gx": 25281856 }
```

| Key | Meaning |
|---|---|
| `x` | The 3-center machinery's aux pair-class engagements |
| `p3` | The screened 3c orbital pair-class engagements |
| `g3` | The 3c ERI kernel weight (the primitive-pair-product sum) |
| `qx` | The nested direct exchange's screened quartet count |
| `gx` | The nested direct exchange's ERI kernel weight |

The exact count moments are defined by the engine instrument
(`ri_engine.hpp` `RiTermCounters`); the vector's `s`/`p` geometry terms
and the `qx_occ` occupancy product are model terms — derived by the fit,
never emitted. The block is **present** exactly when the run was
instrumented: the `ri_j_link` builder with a `[scf] trace_file` set (every
main-SCF BuildFock call then carried the per-call stats sink the exchange
counts ride — the snapshot is complete). Absent on every other run — a
non-ri_j kind or a trace-less ri_j run — never a fabricated block.

### 3.9 `certified_bound`

The certified mixed-precision bound block of a run on the direct
family's **machinery** members: the fp32 lane's accumulated
density-weighted kernel-bound sum, in hartree. The builder computes this
per Fock build (each fp32-routed quartet's kernel bound times its
max-|D| block weight — an upper bound on the Fock-element error that
quartet's fp32 arithmetic delivered, per `fock_build.hpp`); the driver
hands the recorder's per-call reduction to the record instead of
discarding it, which is what this block adds (schema 14):

```json
"certified_bound": { "calls": 20, "last_call_ha": 1.3e-3, "max_call_ha": 2.1e-3,
                     "enforced": false, "budget_ha": 0.0, "routed_ha": 0.0,
                     "routed_quartets": 0, "fell_back_to_fp64": false }
```

| Key | Type | Meaning |
|---|---|---|
| `calls` | int | Main-SCF Fock builds the block summarizes — every build that carried the bound out-parameter, converged or last-iterate. Never 0 in a serialized block |
| `last_call_ha` | number | The final observed call's bound sum: the build whose Fock matrix the reported energy belongs to. This is the per-call quantity a preset sweep's "bound sum" column records |
| `max_call_ha` | number | The largest bound sum over the observed calls — the conservative single-call reading |
| `enforced` | bool | The global budget enforcement ran on the final observed call ([method] enforce_certified_bound, §2.1). False = the key was not set, and the four members below are the enforcement's zero defaults, **not** measurements |
| `budget_ha` | number | The budget the accuracy preset derived for that call: `PresetEnergyBudget(accuracy)` less the call's screened-out bound sum less the ladder's slack (`CertifiedBoundBudget`). `0.0` is a real budget — the screening co-term consumed the preset's target |
| `routed_ha` | number | The final observed call's **routed** bound sum — the routing gate's own a-priori bound units, summed over the quartets the gate admitted. This is the quantity compared against `budget_ha`, and it is **not** `last_call_ha`: that one is the delivered kernel-bound sum, roughly four to five orders larger per routed quartet on the measured fixtures (`certified_budget_test.cpp` prints both) |
| `routed_quartets` | int | The final observed call's **admissions** — the fp32 task list the routing pass produced, i.e. what the comparison was run on, not what survived it. Stays positive on a fall-back; `0` only when the gate admitted nothing |
| `fell_back_to_fp64` | bool | The comparison's verdict on the final observed call: the routed sum exceeded the budget, so the build ran its whole quartet set on the fp64 lane — the same quartet set on the same code path as a lane-disabled build (the direct build is not bit-reproducible, so the two agree to the builder's own run-to-run floor, not bit for bit) |

**The enforcement members are one reading, not two blocks.** They answer
the question the delivered bound cannot: whether the lane was *allowed* to
run. A run at the operating presets reads `enforced: true`,
`routed_ha` ≈ 7.5e-5 against `budget_ha` ≈ 7.7e-11,
`routed_quartets` ≈ 6.1e5 and `fell_back_to_fp64: true`
(`last_call_ha: 0.0`) on the 106-BF C4H10/def2-SVP fixture — the
preset-derived budget refuses the lane by about six orders of magnitude on
the bound sum, which is what the measured enforcement verdict is
(`certified_budget_test.cpp`).
A consumer must therefore never read a zero `last_call_ha` on an enforced
run as "the lane had nothing to do": the block's `fell_back_to_fp64` says
the budget refused it, and `routed_quartets` says how much it refused.

The block is **present** exactly when the seam observed at least one Fock
build that actually carried the bound out-parameter, and **absent**
everywhere it cannot be backed: the direct family's lean member (the
Schwarz-only builder computes no such bound at all — it has no fp32 lane),
the RI-J/QFMM/GPU families (no bound out-parameter), and the UHF branch —
never a fabricated block, and never a fabricated 0.0 that would read as
"certified with zero error". The seam observes *inside* the branch that
selects the out-parameter form (the `requires` test on `BuildFock`), so a
shorter-form builder — one whose `BuildFock` is never handed the pointer,
and which therefore never writes it — cannot contribute an observation at
all: the fields are filled from the builders that reported them and from
no others.

A **present** block whose bound reads `0.0` is a true zero, not an
absence: the machinery builder ran and its certified lane delivered
nothing. That is the expected reading at `accuracy = "kTight"` (the
routing threshold `MixedPrecisionThreshold(kTight)` is 0.0, so the gate
routes no quartet), and it is also the reading on small systems at the
looser presets, where the a-priori gate `C_class·1e-7·Q_bra·Q_ket·dMax`
exceeds the preset threshold for every quartet. The gate is a **routing
criterion, not an error cap** — a `0.0` bound records that the fp32 lane
contributed nothing to this run, never that the run is exact.

### 3.10 `qfmm_model`

The model the composed-QFMM builder's Coulomb half **actually ran**, and the
accuracy the far field was held to (schema 29; the budget keys are schema 37).
Present in a document exactly when the run wired that builder —
`fock_builder = "qfmm"`, however it was reached (the explicit family word or
the no-`fock_builder` size ladder's own tier), on both method types, and on
the legacy null-budget path as well as the budget path. Absent on every other
family, never a fabricated block (the `certified_bound` rule); deliberately a
member of the result rather than of `resources_resolved.mode_record`, which
is absent on exactly one of those paths.

It exists because the builder's **defaults** moved: the owner's ruling of
2026-09-15 made the multipole geometry model the pair's product-distribution
ball and the separation test the surface-to-surface ball, and the 2026-09-20
change returned the separation test to the centre-to-width form — now aimed
at each preset's own angle rather than at one global number. The
product-distribution ball is still the extent default; the midpoint-padded
extent and the surface-ball test are kept in the engine's vocabulary, no
longer defaults. Before the first flip the model a run used was the only one
there had ever been, so no record needed to name it; after it, two runs that
agree on every input key can be two different systems — different objects
under the multipole expansion, different far-field sets, different Fock
matrices. This block is the disclosure rule applied to a default — *honoured,
refused by name, or demoted with the disclosure* — applied where nothing was
refused: the model is not selectable from any input key (§2.3), so it cannot
be requested, refused or dropped. It can only be stated.

| Key | Type | Meaning |
|---|---|---|
| `extent_model` | string | The geometry model the octree and the interaction lists were built over: `"kProductBall"` (the centre and radius of the pair's significant product distribution — the object the multipole expansion is actually of, the default since 2026-09-15) or `"kMidpointBound"` (the conservative midpoint-padded bound, retired as a default and reachable from the engine's vocabulary only). |
| `separation_mode` | string | The well-separatedness test: `"kWidthTheta"` (distance × theta against the nodes' box half-widths — the default, and the form the preset ladder's own angles were derived against) or `"kSurfaceBall"` (surface-to-surface — distance against the nodes' charge radii plus a buffer — the engine's other form, which no input key reaches; `[method] theta` overrides the ANGLE and cannot name a test). |
| `separation_k` | number | The surface-ball buffer, in units of max(rA, rB): 0 the bare touching test, positive the multipole error bound's buffer, **negative the degenerate gate** (nothing well separated — the far field is empty and the build is the exact restricted near-field direct build). Read it under `"kSurfaceBall"` ONLY: the engine records the parameter of the test that RAN and writes 0.0 here under `"kWidthTheta"` — a don't-care of a test that never read it, not a claim that the touching test ran. Every run this driver wires takes the default test, so the block a run document carries names `"kWidthTheta"` and reads 0.0 here; the surface-ball value comes from the engine's own callers. |
| `theta` | number | The resolved theta of the built octree — the centre-to-width parameter, and the angle the build classified with. Read it under `"kWidthTheta"` only, and read it WITH `separation_mode` rather than alone. Three inputs resolve here: `[method] theta` absent is `ThetaForPreset(accuracy)` — the rung's own aim, 0.45 at kLoose, 0.3 at kNormal, 0.0 at kTight; a written theta > 0 is that value; a written theta < 0 is the degenerate gate, 0.0. **theta <= 0 is "never well separated"**: the far field is empty, and `geometric_far_pair_count` reads 0 beside it. The member reads 0.0 under `"kSurfaceBall"`. |
| `error_aware_admission` | bool | Schema 37. Whether the error arm was armed: admission consulted the preset's error budget as well as the separation test, so the interactions the far field admitted are the ones the budget certifies rather than everything the geometry admitted. `false` on every build that did not ask for the arm, which is the default at every preset — and `false` is then a statement about the **arm**, never about the budget: the keys below record the bound's verdict on that build either way. |
| `far_field_budget` | number | Schema 37. The preset's far-field error budget, in hartree — the whole budget the far field was held to, of which `per_interaction_budget` is one interaction's share. |
| `per_interaction_budget` | number | Schema 37. The error share each far interaction was allowed, in hartree: `far_field_budget` split over the geometrically admissible pairs. It reads 0 when there are no far pairs — the absence of a share, not a share of zero. |
| `geometric_far_pair_count` | int | Schema 37. The far field's own interaction count: the pairs the separation test admitted, and the denominator `per_interaction_budget` was split over. It is the **geometric** count — the far field the build would have had without the error arm — so it is unchanged on a run whose arm moved pairs away; `pairs_moved_to_near_field` carries that beside it. |
| `pairs_moved_to_near_field` | int | Schema 37. How many pairs the separation test admitted and the budget refused, computed into the near field instead. 0 on a build that did not arm the error arm, and on an armed build whose budget admitted the whole geometric far field. |
| `fell_through_to_cap` | int | Schema 37. How many far interactions ran at the order cap with their a-priori truncation bound still above their share of the budget — the fall-through that used to be silent. **0 while `error_aware_admission` is `true`**, because that arm admits only what the bound certifies; a nonzero count therefore names a build the bound does **not** certify. It is not the measured error: the bound is a worst-case-over-distributions bound rather than a prediction, so a nonzero count beside an inside-budget energy is the honest record of exactly that and not a contradiction between the two. |
| `worst_truncation_bound` | number | Schema 37. The worst interaction's truncation bound at the order cap, in hartree: the largest bound over the far field, and the number to read `per_interaction_budget` against — **divide it by that share** for how far outside the budget the worst interaction sits. It reads 0 when there are no far pairs: the absence of a worst interaction, never a certified one. |

**The seven budget keys say whether the far field met the accuracy it was
held to** (schema 37), and they exist because the alternative was a document
that could not tell a run whose far field met its budget from one that missed
it. The far field's accuracy control compares each interaction's a-priori
truncation bound against that interaction's share of the preset budget; an
interaction whose bound still exceeds the share **at the order cap** runs at
the cap with the budget missed. Nothing in the record said so before: two runs
whose JSON differed only in the energy were indistinguishable, and a reader
had no way to tell that the model named beside them had run outside its
accuracy. `fell_through_to_cap` is that count and `worst_truncation_bound` is
the size of the worst miss; `per_interaction_budget` is the share to read the
bound against, and `geometric_far_pair_count` the denominator that share was
split over. `pairs_moved_to_near_field` and `error_aware_admission` say
whether the error arm acted on the admission at all. **The keys are written on
every present block**, with the arm's own flag as the answer for the builds
that did not arm it — the `certified_bound` rule: a consumer takes the flag as
the state of the arm rather than inferring it from an absent key, and never
reads an absent budget key as "the budget was met".

**Both words are written, because they are independent facts.** The extent
model and the separation test are separate choices (the engine's vocabulary
holds all four combinations); a record carrying one word would make the other
invisible exactly where the flip moved it. The values keep the engine's
**enumerator vocabulary** — the `mode_record.mode` precedent
(`"kFastPath"`/`"kLightPath"`, not a new lowercase vocabulary) — so a reader
can grep the word a record carries in the source that chose it.

**The values are the engine's, not the driver's.** The block is filled from
`QfmmJBuilder::ModelRecord()`, forwarded through `QfmmHfFockBuilder::
ModelRecord()` (the composed builder is the one the driver holds), because the
theta resolution — the preset's value, the explicit override, and the gate
that every absent-theta run at a gate-preset resolves to — is decided inside
`QfmmJBuilder::Create`. A driver that recomputed it could disagree with the
build it describes, which is the rule the `ri_orbit_expansion` block states
for the same reason.

The states a reader must be able to separate, as the driver tests assert them
(H2/STO-3G, `fock_builder = "qfmm"`; the count column is the far-pair count
that fixture's own geometry admits, and this fixture admits none — the count
is what moves on a chain: the driver's chain runs on the shared alkane
fixture record 0 at kTight (all three chains), 9 (C6) / 2080 (C12) / 73785 (C24)
at kLoose and 0 (C6) / 517 (C12) / 33404 (C24) at kNormal. Four of those
cells are pinned in the driver's chain runs — C6 kTight, C12 kLoose, C24
kLoose, C24 kNormal — each to the number its own run recorded):

| Input | `extent_model` | `separation_mode` | `separation_k` | `theta` | `geometric_far_pair_count` |
|---|---|---|---|---|---|
| `accuracy = "kNormal"`, no `theta` | `kProductBall` | `kWidthTheta` | `0.0` | `0.3` | `0` |
| above + `theta = 0.5` (the angle overridden) | `kProductBall` | `kWidthTheta` | `0.0` | `0.5` | `0` |
| above + `theta = -1.0` (the degenerate gate) | `kProductBall` | `kWidthTheta` | `0.0` | `0.0` | `0` |
| `accuracy = "kTight"`, no `theta` (the rung's own gate) | `kProductBall` | `kWidthTheta` | `0.0` | `0.0` | `0` |

## 4. Units

| Quantity | Unit |
|---|---|
| Energies (`*_energy_hartree`, `rms_error`, all `nocv` energies) | hartree |
| Coordinates in the input file | Ångström by default; **`[molecule] units` names the unit** (schema 28), and the parser converts to Bohr ONCE at the file boundary. Every coordinate the record reports (QTAIM, `density_at_nuclei`) is Bohr |
| Dipole | e·a0 |
| Quadrupole | e·a0² |
| Populations, charges, valences, ESP fit charges, Fukui, EDDB, NOCV eigenvalues | electrons |
| Density at the nuclei (`density_at_nuclei.values`), QTAIM `density` | electrons/bohr³ |
| QTAIM `laplacian`, `eigenvalues` | electrons/bohr⁵ |
| `timings_ms` | milliseconds, wall clock |

## 5. Limits and unsupported combinations

The errors below are what a user actually sees when a limit is tripped.

**Element-count limits:**

- **Hirshfeld and the `sad` guess: Z <= 10.** The promolecular fragment
  densities and the SAD guess come from the same atomic-multiplicity table:
  `the SAD atomic-multiplicity table (v1, Z <= 10) does not cover element Z = <N>`
- **Nalewajski-Mrozek and NOCV: Z <= 19.** Both use the Aufbau
  shell-degeneracy table:
  `the Nalewajski-Mrozek Aufbau table covers Z = 1..19` /
  `the Aufbau table covers Z = 1..19`

**Method limits (v1, rejected before any computation):**

- `ri_jk` fock builder on the UNRESTRICTED leg, and on the Kohn-Sham lanes —
  OPENED on both (the unrestricted leg with its per-spin adapter over the pair
  entry point; the Kohn-Sham lanes 2026-09-16, the composition over that same
  pair, `MakeRiFullKsHalf`: H added to both halves and the factor of two onto
  the Coulomb one). What still refuses on those lanes is `gpu`/`gpu_split` and
  the `ri_j_link` disk rungs (the KS lanes gained `ri_j_link` 2026-09-16;
  `qfmm` GAINED the KS lanes 2026-09-16). This line read "`ri_jk`/`gpu` and
  the `ri_j_link` disk rungs still refuse" until 2026-09-16, and the `ri_jk`
  half of it was stale — verified by running the cell: H2O/STO-3G, RKS and
  UKS, `fock_builder = "ri_jk"`, exit 0, converged. **The closed-shell `ri_jk`
  path is WIRED and runs** (`integrals RiFullFockBuilder`, the composed `F = H
  + 2 J_RI - K_RI` with no nested direct-exchange call); the line here read
  "no full-RI exchange path exists" until 2026-09-15, and it was stale —
  verified by running it: H2/STO-3G, `fock_builder = "ri_jk"`, exit 0,
  converged, E = −1.116738328800222. **On this path `[method] accuracy`
  governs the SCREENING CO-TERM ONLY**: the preset's budgets bound the
  screening truncation, while the run's error against exact K is the error of
  the auxiliary fit named in
  `resources_resolved.selection.approximation.aux_basis` — tightening the
  preset does not improve that fit, and no reading of this key may infer
  otherwise. **What that key DOES stage on this path is the path's own preset
  mapping, stated once in code**
  (`integrals/include/qcx/integrals/accuracy.hpp`): the screening thresholds
  are the preset's `SchwarzThreshold` and — once the auxiliary-pair density
  screen lands — its `DensityThreshold`, never a second table; the auxiliary
  QUALITY is **not a preset knob**, because `RiExchangeRequiresJkFit` returns
  true at all three presets (a JK-optimized fit is required at every rung, and
  a J-only fit is refused outright), so `accuracy` may never be read as a
  request for a better or a coarser fit; and the per-atom error the path is
  measured against is the preset's own `RiExchangeErrorBudgetPerAtom`. The
  measurement against that bar is taken per preset, both sides at the same
  one, in that validation cell (`integrals/tests/ri_jk_validation_test.cpp`).
  **Since schema 31 that measurement reaches the run record**: an
  approximated-exchange run emits the achieved per-atom error beside this
  run's preset bar in
  `resources_resolved.selection.approximation.exchange_error` (§3.6), with the
  fixture it was measured on, so a reader of an energy this path produced can
  size it from the JSON alone. The value's own home is
  `RiExchangeWorstMeasuredPerAtomError`, in this same mapping file.
- UHF with any builder but `direct`.
- RHF with a `sad` guess (the SAD fragment inputs are UHF-only in v1). RHF
  accepts `core`, `gwh`, and `restart`; `core` and `gwh` take the RHF loop's
  default start, the GWH guess — `core` means "the method's default start",
  not the zero matrix. `restart` is the run-restart read (seeds the
  last-iterate density plus DIIS history); it replaces the guess entirely, so
  a resumed run's first iteration is the saved iterate's continuation.
- UHF with `guess restart` or `scf.checkpoint_file` — both the restart read
  and write are RHF-only in v1 (the RHF solver is the one whose options carry
  the `ScfRestartState` seed); UHF supports guess `core`, `gwh`, and `sad`.
- RHF `guess restart` of a checkpoint that is not an RHF store (UHF store,
  or a store of another molecule/basis — the fingerprint group check), or
  of a missing/corrupt file: refused before the SCF, never a silent
  fallback to the GWH start.
- ETS-NOCV on a UHF run (`nocv_fragments` + UHF) — the decomposition's
  per-spin resolution assumes the closed-shell D_sigma = D/2; a UHF
  molecular density would silently feed it the wrong spin channels, so the
  combination is rejected up front.
- A `nocv_fragments` index outside the file's atom rows.
- `enforce_certified_bound` on a route that cannot honour it — the
  LightPath chunk loop (`the certified-bound budget enforcement does
  not compose with the LightPath chunk loop yet`) and an engaged precision
  ladder (`... does not compose with an engaged precision ladder`). The key
  is never silently dropped: a run whose route cannot run the comparison
  fails with `kUnimplemented` instead of computing as if it were absent.
  The lean member, the RI-J/QFMM/GPU families and the UHF branch do not
  consume the key at all — none of them has an fp32 lane to enforce, and
  none of them reports a `certified_bound` block.
- `force_certified_lane` on a route that runs no reachable fp32 lane — the
  direct family's **lean** member (the default at `nBasis <= 1000`, whose
  builder has no fp32 lane and no such field), and the RI-J/QFMM/GPU
  families (whose own lanes do not read the request). The key is never
  silently dropped: the run fails with `kUnimplemented` naming the route
  and the remedy instead of computing as if it were absent. Unlike
  `enforce_certified_bound` above, the refusal also fires at
  `accuracy = "kTight"` on the routes that COULD honour it: that preset's
  `MixedPrecisionThreshold` is 0.0 by construction, so the lane is off
  there whatever the request says — accepting the key would be the silent
  no-op the enforcement contract forbids.

## 6. Worked example

The input below is the H2 STO-3G fixture of `driver/tests/run_driver_test.cpp`
(`kH2Toml`); the JSON is the real output of `qcx.exe run` on the revision this
page describes (timings vary by machine):

```toml
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["H", 0.0, 0.0, 0.0],
    ["H", 0.74084809526419992, 0.0, 0.0],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kTight"

[scf]
max_iterations = 100
energy_tolerance = 1e-8
density_tolerance = 1e-6
use_diis = true
```

```json
{
  "converged": true,
  "electronic_energy_hartree": -1.8310000394614827,
  "iterations": 2,
  "moments": {
    "dipole": [
      4.440892098500626e-16,
      0.0,
      0.0
    ],
    "quadrupole": [
      [
        0.6258891072758024,
        0.0,
        0.0
      ],
      [
        0.0,
        -0.31294455363790163,
        0.0
      ],
      [
        0.0,
        0.0,
        -0.31294455363790163
      ]
    ]
  },
  "populations": {
    "gopinathan_jug": {
      "bond_orders": [
        [
          0.4999999999999991,
          0.4999999999999991
        ],
        [
          0.4999999999999991,
          0.4999999999999991
        ]
      ]
    },
    "lowdin": {
      "alpha": [
        0.49999999999999956,
        0.49999999999999956
      ],
      "beta": [
        0.49999999999999956,
        0.49999999999999956
      ],
      "spin": [
        0.0,
        0.0
      ],
      "total": [
        0.9999999999999991,
        0.9999999999999991
      ]
    },
    "mayer": {
      "bond_orders": [
        [
          0.0,
          0.9999999999999996
        ],
        [
          0.9999999999999996,
          0.0
        ]
      ],
      "free_valences": [
        0.0,
        0.0
      ],
      "total_valences": [
        0.9999999999999996,
        0.9999999999999996
      ]
    },
    "mulliken": {
      "alpha": [
        0.4999999999999999,
        0.4999999999999999
      ],
      "beta": [
        0.4999999999999999,
        0.4999999999999999
      ],
      "spin": [
        0.0,
        0.0
      ],
      "total": [
        0.9999999999999998,
        0.9999999999999998
      ]
    }
  },
  "schema_version": 1,
  "spin_squared": null,
  "timings_ms": {
    "scf_loop": 0.839,
    "total": 21.0072
  },
  "total_energy_hartree": -1.1167143251757685
}
```

What to notice in this output:

- `spin_squared` is present with value `null` — the RHF case (§3.1). The
  O2 UHF pin below shows the value case.
- `mulliken`/`lowdin` totals are exactly 1.0 per hydrogen and the spin
  arrays are identically zero — the RHF convention (§3.2), not a bug.
- `mayer.free_valences` is zero — RHF by construction.
- The dipole is the zero vector to within floating-point residue (the
  x-component is ~4.4e-16, the pin test asserts 1e-7); the quadrupole is
  present, nonzero, and traceless (0.6259... − 2 × 0.3129... = 0) — both
  moments are always emitted (§3.3).
- `gopinathan_jug.bond_orders` carries the diagonal (0.4999... everywhere)
  — the G-J matrix includes the diagonal, unlike `mayer.bond_orders`.
- No opt-in analysis key (`charges`, `esp`, `eddb`, `fukui`, `nalewajski`,
  `nocv`, `density_at_nuclei`, `qtaim`) appears anywhere — none was
  requested (§3.4).
- The document was captured at `schema_version: 1` (pre-`density_at_nuclei`); current runs carry version 37.

### Other pin fixtures

| Run | Pin values |
|---|---|
| H2O STO-3G RHF (`kH2oToml`) | total −74.96292827; dipole y 0.67898079210 e·a0; O mulliken total 8.3663559645 |
| O2 STO-3G UHF SAD (`kO2Toml`) | total −147.63394678545018; `spin_squared` 2.0034108576810308; mulliken spin [1.0, 1.0] (one unpaired electron per oxygen); dipole zero by symmetry |

## 7. Versioning policy

`RunResult::kSchemaVersion` (serialized as `schema_version`) is bumped by 1
on any change a JSON consumer must notice: a key added or removed, a unit
changed, a nullability rule changed. The bump, this document's update, and
the serializer tests' update land in the same change. Consumers should key
off `schema_version` rather than any per-key heuristics.

**A word that canonicalizes is not a bump.** An accepted INPUT alias — a
spelling a consumer can never observe, because it resolves to a word that
already existed and the record carries that word — changes nothing a JSON
consumer must notice, so it takes no number. The tier axis's `in_memory`
(2026-09-15, the owner's two-axis ruling) is that case, and it took **no
number**: it is an alias of `direct` (`[method]
fock_builder` in §2.1), both spellings fill `method.builder` with
`kDirect`, and no record field a consumer keys on can differ between them —
the builder, member, explicit-request and lean fields are identical, and
only the measured wall times vary between runs, as they do between two runs
of one input. A change is a bump exactly when the record can differ, which
an alias by construction cannot.

**Version history:**
- 1 → 2 (2026-08-29, the first analysis increment): the
  `density_at_nuclei` input key and output block added (the `values` member,
  per atom, electrons/bohr³).
- 2 → 3 (2026-08-29, the resource caps): the `[resources]` input block
  and the `resources_resolved` output block added (the memory cap and
  thread ceiling the run actually executed under, plus whether the hard
  in-process cap was applied).
- 3 → 4 (2026-08-29, the resource caps' fourth step): the `memory_model` audit
  block added to `resources_resolved` (the calibrated per-builder modeled peak
  the caps were decided against). A run the seam refuses carries the same
  terms in its error diagnostic. (The block was REMOVED at 35 → 36, 2026-09-17
  — the entry is kept as the record of what happened.)
- 4 → 5 (2026-08-29, the second analysis increment): the `qtaim` input key and
  output block added (the (3,-1) bond critical points of rho with their
  bond paths, ellipticity and laplacian, the reported non-bond /
  truncated-path critical points, and the unconverged Newton seeds).
- 5 → 6 (2026-08-30, the molden increment): the `[properties] molden`
  input key and the `molden` output block added (the path of the written
  Molden-format F-file, absent when not requested).
- 6 → 7 (2026-08-30, GPU auto-selection): `[method] fock_builder` became
  optional (absent = the heuristic auto-selection) and the
  `resources_resolved.selection` record added (the wired builder, the
  heuristic pick, the reasoning, the explicit request, and the divergence
  warning; the record serializes last in the block).
- 7 → 8 (2026-08-30, full-group labeling + the workspace budget): the
  `[symmetry] full_group` input key (default true) and the `symmetry` /
  `symmetry_beta` output blocks added (the per-MO labels, the detected full
  group and Abelian reduction, the canonicalization/straddle records, the
  symmetrization subset); the `resources_resolved` `workspace_budget` and
  `mode_record` records added (the budget path: the cap-minus-base capacity,
  the cumulative Create-time commit, and the engine's rung decision with its
  firing estimate terms). The class-table admission records
  (`class_table_bytes`, `class_path_disengaged`) and the UHF exchange half's
  `exchange_mode_record` (same shape as `mode_record`) are folded into this
  entry — version 8 never shipped without them. The `[resources]` block's
  `memory_cap_gib = 0` escape hatch still yields the legacy null-budget path,
  whose documents carry none of the new records.
- 8 → 9 (2026-09-02, the RI-J emission lane): the
  `term_counters` output block added (§3.8) — the engine-emitted
  x/p3/g3/qx/gx per-run totals of an instrumented `ri_j_link` run (an
  `[scf] trace_file` set); absent on every other run.
- 9 → 10 (2026-09-03, the disk-rung knob):
  the `[method] ri_tensor_mode` and `[method] ri_chunk_bytes` input keys
  added (the disk-rung selector and the chunk-size passthrough); the
  `mode_record` gained the `disk_bytes` member (the disk rung's modeled
  dense on-disk payload, zero on in-memory records) and `mode` may now be
  `"kDisk"` — the driver-synthesized record of the disk route (no engine
  sets the mode or the term; integrals cannot link storage).
- 10 → 11 (2026-09-04, the selection-coefficients provenance):
  the `resources_resolved.selection` record gained the `file_version`,
  `machine_class_key`, `basis_family_id`, `preset_id`, `borrowed` and
  `scores` members — the per-candidate `{candidate, score_seconds, source,
  selectable}` objects, the optional `cell_key`, and the inflation marker
  of the coefficients provenance (the exact candidates and scores the
  heuristic weighed). (This entry was never written to the history at the
  bump — repaired 2026-09-07 together with the 11 → 12 bump.)
- 11 → 12 (2026-09-07, the k and team read): the
  `mode_record` / `exchange_mode_record` objects gained `concurrent_slots`
  (the Create-time authorized k of the bounded-concurrency
  batch loop) and `default_team_size` (the decision's clamp-origin
  `DefaultOmpTeamSize()` read; `concurrent_slots == default_team_size`
  marks a team-clamped run, the k=5 reading of that diagnosis). A
  consumer that reports on which k a fired run carried must notice.
- 12 → 13 (2026-09-10, the lean flag): the
  `memory_model` object gained the `lean` boolean — the direct family's
  within-family lean member is a record flag, never a new builder-slot
  word (the lean ruling: the family-level "direct" word stays). A
  consumer that reports which family member a run actually executed must
  notice: `builder` alone cannot tell the machinery from the lean member,
  and on runs predating the flag the choice is not recoverable from the
  record. (The io half landed the member, the serializer key and the
  bump; this entry, the driver plumbing that sets it, and the driver
  tests landed in the completing change, per the policy above.) (The
  `lean` flag and the `memory_model` object that carried it were REMOVED at
  35 → 36, 2026-09-17; the choice is read from `selection.builder_member`.)
- 13 → 14 (2026-09-11, the certified-bound observability): the
  `certified_bound` output block added (§3.9) — the fp32 lane's
  accumulated density-weighted kernel-bound sum, the direct builder's
  `certifiedBoundSumOut` out-parameter, which the driver had computed and
  then discarded (it passed `nullptr` on every production run). A
  consumer that reports on certified mixed precision, or that attributes
  an energy difference to the fp32 lane, must notice: the quantity was
  previously unrecoverable from any run record. The block is absent — not
  zero — wherever the builder computes no such bound (the lean member,
  the RI-J/QFMM/GPU families, the UHF branch). (The io half landed the
  member, the serializer key and the bump; the driver plumbing that fills
  it and the end-to-end tests landed in the completing change, per the
  policy above.)
- 14 → 15 (2026-09-11, the selection member name): the
  `resources_resolved.selection` record gained the `builder_member` string
  — the within-family MEMBER the run wired (`"lean"` for the direct
  family's lean Schwarz-only member, the family word itself everywhere
  else). Additive by ruling (user, 2026-09-11): `selection.builder` KEEPS
  reporting the family word (`"direct"` on a lean run), every existing key
  is byte-identical, and there is no migration. A consumer that reports
  which builder produced an energy must notice: before this key the lean
  member was only INFERABLE (from an absent `explicit_builder` at
  `n_basis <= 1000` — a proxy that is also absent for the explicit lean
  word, and ambiguous above the ceiling), and only the `memory_model.lean`
  flag recorded the choice at all. (That flag was REMOVED at 35 → 36,
  2026-09-17: `builder_member` above is now the one record of the choice,
  and it names the member rather than flagging it.)
- 15 → 16 (2026-09-12, the global budget enforcement): the `[method]
  enforce_certified_bound` input key added (§2.1) and the `certified_bound`
  output block gained the enforcement's own members (§3.9) — `enforced`,
  `budget_ha`, `routed_ha`, `routed_quartets`, `fell_back_to_fp64`, all
  measured on the same final observed call as `last_call_ha`. A consumer that
  reads `last_call_ha` or `max_call_ha` as the lane's delivered bound must
  notice: `routed_ha` is a DIFFERENT quantity (the routing gate's own units,
  the one compared against the preset-derived budget, ~5 orders smaller per
  routed quartet on the measured fixtures), and `fell_back_to_fp64: true` with
  a zero `last_call_ha` is a lane that delivered nothing BECAUSE the budget
  refused it — not a lane with nothing to deliver (`routed_quartets` stays
  POSITIVE on a fall-back: it counts the admissions the comparison ran on, and
  reads 0 only when the gate admitted nothing). The members are present
  (false/zero) on every block, including an unenforced run's, so the reader
  never infers the check's state from an absent key. Off by default: the
  enforcement moves quartets from the fp32 lane to fp64, so it is a behaviour
  change an explicit builder request must not take silently.
- 16 → 17 (2026-09-12, the SCF convergence-flag truthfulness lane): the
  top-level `energy_delta_hartree` and `rms_density_delta` keys added
  (§3.1) — the gate's OWN operands at the returned iterate, recorded to
  qualify `converged`, which is a bare Boolean that does not say what it
  stood on. A consumer that treats `converged: true` as "this energy is
  within the requested tolerance" must notice: on the RHF path the gate's
  energy leg is inert (the note in `rhf.cpp` — the loop overwrites its
  previous-iteration energy before the gate reads it), so the stop is
  decided by `[scf] density_tolerance` ALONE and
  `energy_delta_hartree` is not bounded by `[scf] energy_tolerance`. On
  the measured C4H10/def2-SVP fixture at the shipped defaults the run
  stops at 9 iterations with `energy_delta_hartree` 2.29e-5 against an
  `energy_tolerance` of 1e-8 — and 3.6163e-6 Ha from the same binary's
  own 1e-10 answer. The sign is the LOWER of the two, not above: the
  reported energy is 1/2 Tr[D (H + F)] with the DIIS-extrapolated Fock,
  not the Fock of the returned density, so at finite convergence it is not
  a variational upper bound. `[scf] energy_tolerance` is inert on RHF for every
  value (1e-3 and 1e-10 give bit-identical results), not merely loose;
  the option is retained, validated and documented as inert at
  `RhfOptions::energyTolerance`, and making the leg live would move every
  existing RHF trajectory — a behaviour change the deferral owns.
  Both keys are always present, `null` when no SCF loop filled them
  (never a fabricated 0.0).
- 17 → 18 (2026-09-12, the forced disk-rung mode): the `[method]
  ri_tensor_mode` input key gained the `forced_disk` value (§2.1) and the
  `mode_record` / `exchange_mode_record` objects gained `forced_disk` —
  the forced diagnostic route's own statement, `true` only where the
  driver bypassed the composed in-memory ladder and constructed the
  storage-module disk builder directly (and `false` on every engine rung
  decision and on the ladder's own disk fallback). The `mode` vocabulary
  is unchanged (`kDisk` stays the one disk word) and `disk` keeps its
  rung semantics (the ladder's LAST rung; the last-rung rule and the
  null-budget-path refusal are untouched), so this bump exists for the
  one thing a consumer must notice: a `mode_record` with
  `mode: "kDisk"` no longer implies the ladder sent the run there. A
  consumer that reads a disk cell as "the ladder refused and disk was the
  fallback" must read `forced_disk`: on `true` no ladder ran at all, so
  the record carries no decision terms and the cell measured the disk
  builder because it was asked to.
- 20 → 21 (2026-09-13, the certified lane's force-on input, the owner's ruling
  2026-09-13): the `[method] force_certified_lane` input key (§2.1) and the
  `resources_resolved.compute_profile.certified_lane_forced` member (§3.6) —
  the supported, recorded way to engage the certified fp32 lane whatever the
  run's compute-profile probe measured, which until this landed was reachable
  only by injecting a probe's return value in source. A consumer that reads
  `certified_lane_default` as the lane's STATE must notice: the block keeps
  publishing the machine's verdict unchanged, and `certified_lane_forced`
  says when the lane's state did not come from it (a forced run reads
  `certified_lane_forced: true` beside `certified_lane_default: false` on a
  machine whose probe resolves OFF). Additive in the block — no existing key
  changed shape or nullability — but not silent in the input schema: a
  `force_certified_lane = true` run on a route that cannot carry the request
  (the direct family's lean member, the RI-J/QFMM/GPU families, and
  `accuracy = "kTight"` on every route) now fails with `kUnimplemented`
  rather than computing as if the key were absent (§5).
- 19 → 20 (2026-09-13, the certified lane's decided record):
  the `resources_resolved.compute_profile` block added (§3.6) — the measured
  fp32/fp64 FMA-throughput ratio the certified fp32 lane's default was
  resolved against, with the parameter `source` (`"host"` | `"device"`), both
  GFLOP/s, the `measured` flag, the pair count and spread, the verdict, and
  the threshold it was compared against. It exists because the lane's default
  became HARDWARE-AWARE and was otherwise unrecorded: a run reported
  what the lane DELIVERED (`certified_bound`) but never what decided it, so a
  verdict could not be checked against the machine that produced it. A
  consumer that assumed the lane's default came from a fixed switch, or that
  a 1.0 ratio meant "no probe", must notice: `measured` separates a scalar
  machine's real 1.0 from the documented unknown, and `source` says which
  arm answered. Additive — no existing key changed shape or nullability.
- 18 → 19 (2026-09-12, the requested-vs-ran pairing): the
  `resources_resolved.ri_tensor_mode` block added (§3.6) — what `[method]
  ri_tensor_mode` ASKED FOR and what actually ran, with the four outcome words
  `honoured` / `ladder_fit` / `not_applicable` / `demoted`, and a `reason`
  when the two do not agree. It exists because the mode record cannot carry
  this: `mode_record` is absent on exactly the paths where a request would
  otherwise leave no trace (the lean arm and the legacy null-budget path), and
  a family with no RI disk rung never had a record for the question at all. A
  consumer that reads `mode` as "what was asked for" must notice: `mode` names
  what RAN, and the request side now has its own block, which is also what
  makes a demotion legible — under the disclosure rule's demotion (not
  refusal) is the expected outcome for a request the run cannot honour, and
  the record must show both sides for that to be a demotion rather than a
  substitution. `ladder_fit` is deliberately NOT a demotion: `disk` permits
  the ladder's last rung without demanding it, so a fitting memory rung riding
  is the request's own prescription.
- 21 → 22 (2026-09-13, the disk force leaves the rung selector, the owner's
  ruling 2026-09-13): the `[method] ri_tensor_mode` key LOST its `forced_disk`
  word (§2.1), the `[diagnostics]` block and its `force_disk_ri` key were
  added (§2.1), and `resources_resolved.ri_tensor_mode` gained the `forced`
  member (§3.6). The force is the same mechanism it always was — the driver
  still bypasses the composed in-memory ladder and constructs the
  storage-module disk-backed builder directly — and nothing numerical moves;
  what changed is that the force stopped being a value inside a selector. A
  rung word re-used to mean "force" is one key naming two mechanisms, which is
  the conflation the key-split rule separates (a key naming a mechanism is
  honoured, refused or demoted BY NAME; a preference may be disclosed). A
  consumer must notice three things. **(1)** `mode_record.forced_disk` is
  unchanged in meaning and in name — `true` exactly on the forced route — but
  its SOURCE is now the new key, so a cell that used to be staged as
  `ri_tensor_mode = "forced_disk"` must be staged as `[diagnostics]
  force_disk_ri = true`. **(2)** The rung word is refused by name rather than
  dropped, with the key that replaced it as the remedy (§2.3), because a
  capability that moved must not read as one that is gone; the word survives
  in `resources_resolved.ri_tensor_mode.requested`, where it names the request
  rather than the key. **(3)** That block's `forced` member is the only place
  the request's SOURCE is recorded: the same word is no longer reachable from
  the input, so a reader must be able to tell a forced request from a rung
  word without the input file. Additive in the record, and not silent in the
  input schema.
- 22 → 23 (2026-09-13, the `ri_chunk_bytes` disclosure, the owner's ruling
  2026-09-13): the `resources_resolved.ri_chunk_bytes` block added (§3.6) —
  the requested size, whether the hint was honoured or dropped, and why. The
  key itself is unchanged, and its drop stays non-fatal (the key-split rule:
  refusing a size hint on a family with no chunking to size is pedantic); what
  changed is that the drop is no longer silent. The hint's only consumer is
  the `ri_j_link` family's disk-rung construction, and until this change a run
  that named the key and never read it said nothing about it anywhere in the
  document. A consumer must notice that the new block's `outcome` vocabulary
  is NOT its sibling's: `ladder_fit` on `ri_tensor_mode` means the rung
  request was not lost, while the same run DROPS the chunk-size hint, so the
  two blocks make opposite statements about one run and each is true of its
  own key. Additive — no existing key changed shape or nullability.

- 23 → 24 (2026-09-13, the exchange-approximation disclosure): the
  `resources_resolved.selection.approximation` block added (§3.6) — the
  contraction form the exchange half was built in, and the auxiliary basis
  name in effect for it. Written from the `ri_jk` run path that the same
  change wired. A run whose exchange half is an auxiliary fit must say so, or
  a consumer reads an energy built from that fit as though it came from exact
  quartets — the different-result-class problem that disclosure answers rather
  than deletion. **This entry is written by a later change, not by its own:**
  the change that landed the block bumped the constant and moved the
  serializer pin but did not reach this document, and the versioning policy
  above requires all three in the same change. The facts are that commit's;
  the attribution is recorded here so the history does not jump 23 → 25 and
  read as though the block never happened.
- 24 → 25 (2026-09-13, the aux-selection notice): the
  `approximation` block gained **`aux_notice`** (§3.6) — why the auxiliary basis
  was chosen by a policy that warns for the region it is in (minimal bases,
  bases without polarization, diffuse-augmented bases, ECP/relativistic bases,
  and a JK request with no matched JK fit). The aux-selection rule now always
  resolves a default (it refuses nothing outside a singular fit matrix or an
  unmeetable tolerance), so a run in a weak region is demoted-to-a-default WITH
  disclosure rather than refused, and this key is that disclosure. Two
  consequences a consumer must notice. First, `aux_notice` is **absent rather
  than null** when the selection is in no weak region. Second and larger,
  **the block's presence rule is WIDENED**: it now reads "present when the run
  wired an approximate exchange builder **or** the aux selection carries a
  notice", so a run whose kernels are exact still emits the block when the aux
  selection warns. The previous rule ("present exactly when the run wired such a
  builder; absent on every run whose kernels are exact") is superseded, because
  a notice hanging off an ABSENT block is invisible — a reader who sees a
  warning must not have to infer that the block was omitted for the
  exact-kernel reason. On that case `exchange` reads `"exact"`: the meaning the
  absent case used to carry, stated rather than left for the reader to
  reconstruct. Additive, plus the one presence-rule widening — which is the
  change a consumer keyed on the old rule will notice.

- 25 → 26 (2026-09-14, the RI-J orbit expansion reachable, the orbit-expansion
  wiring): `resources_resolved` gained **`ri_orbit_expansion`** (§3.6's
  sibling) and the input gained **`[method] ri_orbit_expansion`**. Additive on
  both sides, with one presence rule of its own: the block is written exactly
  when the input NAMED the key, on every family, because the key's one
  consumer is the `ri_j_link` arm and a request that lands elsewhere must
  still be answerable. The four outcome words are `engaged` (the driver built
  both reductions, the engine reports the expansion ran — read from
  `RiJkFockBuilder::OrbitExpansionEngaged`, never recomputed in the driver),
  `not_requested` (named and false), `inert_trivial_group` (requested, both
  reductions built, and a group order of 1 makes every cell its own orbit),
  and `not_applicable` (requested on a family whose wiring does not consume
  the key). A consumer that reads the key's absence as "the mechanism is off"
  is right; a consumer that reads the ENGINE option's default (ON, a
  permission) as "every run expands" is wrong — the driver supplies no
  reduction unless this key asks **Correction, 2026-09-12 — the inert leg was
  closed (the energy-leg deferral).** The **16 → 17** entry (2026-09-12, the
  SCF convergence-flag truthfulness lane — the one that added the top-level
  `energy_delta_hartree` and `rms_density_delta` keys) describes the state
  that the schema-17 keys were added to *observe*; The change of that date
  ("the RHF energy leg is live, energy_tolerance stops being inert") moved
  `RunScfLoop`'s energy bookkeeping after the gate, so **that entry's
  inertness claims no longer hold** — its "on the RHF path the gate's energy
  leg is inert" and its "`[scf] energy_tolerance` is inert on RHF for every
  value (1e-3 and 1e-10 give bit-identical results)" — including its
  cross-reference to the option being "documented as inert" at
  `RhfOptions::energyTolerance`, whose doc block now reads live. Same day,
  same shape of change, no schema version bump: no key was added, removed or
  re-nullable — the *numerics* moved, the same shape as the 2026-08-30 default
  re-pin (which moved every RHF energy and carried no bump either).

- `[scf] energy_tolerance` is now **live on RHF**. The stop requires both legs,
  and a hidden input no longer has one of them switched off.
- `energy_delta_hartree` **is** bounded by `[scf] energy_tolerance` on a
  converged run, on both paths. The schema-17 keys did their job: they are what
  made the inert leg visible in the record, and they are now simply a true
  report of the criterion that fired.
- **Every cached RHF result from before this change is not comparable with one
  after it.** Measured on this tree, before → after: C4H10/def2-SVP at the
  shipped defaults 9 → 15 iterations, −157.18631939283310 → −157.18631577745569
  (+3.615e-06 Ha); H2O/cc-pVDZ 9 → 13, −76.02679793958652 → −76.02679868147290
  (−7.419e-07); H2O/STO-3G 7 → 8, −74.96292827052788 → −74.96292827063873
  (−1.108e-10). At the tight 1e-10/1e-10 gate the *density* leg is the loose one,
  so the counts move further — 15 → 29 and 16 → 31 — while the energies move only
  −7.9e-09 and −8.3e-09. **Iteration counts move even where energies do not**, so
  a stored count is the least portable thing to cache.
- **No tolerance value changed.** The defaults stay `energy_tolerance = 1e-8`,
  `density_tolerance = 1e-6`; a run that wants the old stopping behaviour
  sets `density_tolerance` explicitly and accepts the energy tolerance being
  real.
- 26 → 27 (2026-09-14, the linear-dependence removal, the owner's ruling): the
  top-level **`num_removed_overlap_directions`** key added (§3.1). The
  orthogonalizer no longer ABORTS on a near-singular overlap: it removes the
  directions below the relative eigenvalue floor, runs the SCF in the reduced
  orthonormal space and maps back (`C_AO = X C_orth`, `D_AO = X D_orth X^T`,
  both in the full AO dimension), and this key is the disclosure the rule requires —
  a silent truncation would otherwise be indistinguishable from a correct run.
  It is a SYSTEM property (basis and molecule), so RHF and UHF report the SAME
  count; 0 is the well-conditioned answer; **null means no SCF loop filled it**,
  never a fabricated zero. The refusal survives only where it is a genuine
  error: fewer usable directions than the system's occupied orbitals. Measured
  on this tree: aug-cc-pVTZ on C24H50 sits at s_min/s_max = 4.92e-09 with ONE
  direction below the floor — before this change that 74-atom alkane in a
  diffuse basis did not run at all.
- 27 → 28 (2026-09-15, the explicit coordinate unit, the owner's ruling): the
  **`[molecule] units`** input key and the top-level **`molecule_units`** record
  key added (§2.1, §3.1) — the unit the atom rows are written in, resolved at
  ONE place (the parser, `ParseAtom`) with the conversion applied there and
  nowhere else, and the resolved value published in every record. A consumer
  must notice the record key because it is new in every document; the INPUT key
  is **purely additive and changes no existing file's meaning** — absent is
  Ångström, which is what the atoms rows always were, and the internal geometry
  is Bohr under either value, so every run that does not name the key produces
  byte-identical numbers to before. What changed for a writer is that the
  convention is now **stated instead of assumed**: a file whose numbers are Bohr
  (`units = "bohr"`) used to be read as Ångström, silently, at 1.89× the
  intended distances — the 2026-09-15 hunt, where a probe fed Bohr numbers to
  the Ångström key and the defect search chased a stretched molecule. An unknown
  unit word is refused by name with the accepted list rather than falling back
  to Ångström (a freely-interpreted typo is the silent default this key exists
  to remove), and the same discipline applies to the record half: the driver
  copies the parser's resolved member, so a record cannot disagree with the
  parse it came from.
- 28 → 29 (2026-09-15, the QFMM model disclosure, the obligation the default
  flip created): the top-level **`qfmm_model`** record key added (§3.10) —
  which geometry model and which separation test the composed-QFMM builder's
  Coulomb half actually ran, with the separation buffer and the resolved
  theta. A key added, so a consumer must notice; and it answers a question the
  record could not answer before, which is why it exists: the same day's owner
  ruling flipped `QfmmOptions`' defaults (product-distribution extents and the
  surface-ball test became the default, the midpoint/width pair retired but
  selectable), so a run's model stopped being readable off any input key — no
  key names it — and two runs agreeing on every key could be two different
  systems. **The input side gains NO key**: the block rides `fock_builder =
  "qfmm"` (explicit or ladder-resolved), it is absent on every other family,
  and its members come from the ENGINE's own `QfmmJBuilder::ModelRecord()`
  through `QfmmHfFockBuilder::ModelRecord()` rather than from a driver-side
  recomputation. The one read rule a consumer must carry is in §3.10: each
  parameter member carries the value of the mode that ran and 0.0 under the
  other, so `separation_mode` decides which of the two to read.
- 29 → 30 (2026-09-16, the RI-K memory ladder's grant): the
  `resources_resolved` **`ri_jk_mode`** block added (`RunRiJkMode`) — the
  composed full-RI builder's Create-time rung record: the rung that fired
  (`kFast` / the batched `kBlocked`), the estimates's byte decomposition, and
  the budget it was decided against. It is a **separate block from
  `mode_record`**, not a member of it, because the two families' rung words and
  term decompositions are different allocations; written only when a decision
  was actually made (the no-budget path consults no budget and takes the fast
  rung, so the block is ABSENT there rather than reporting a decision that did
  not happen). The input side gains no key: the block rides the existing
  `[resources]` grant. Consumer rule: absent means "no decision", never "the
  default rung fired".
- 30 → 31 (2026-09-16, the exchange disclosure's record half): the
  `resources_resolved.selection.approximation` block gained **`exchange_error`**
  (§3.6) — the approximated-exchange path's MEASURED per-atom error with the bar
  the run is read against (`measured_per_atom_hartree`, `bar_per_atom_hartree`,
  `bar_preset`, `measured_on`, `measured_aux_basis`). Keys added, so a consumer
  must notice; no existing key moved, was removed, or changed nullability. It
  closes the gap the owner's 2026-09-16 ruling turned on — the error was
  measured and module-level, but no run could EMIT it, and a disclosure a run
  cannot emit is not a disclosure. Two rules a consumer keys on: the field is
  present **only when `exchange` names an approximated contraction** (the block
  itself is wider since schema 25 — a notice may ride an exact-kernel run, and
  there the fitted error is not that run's), and the measured value is the
  PATH's worst characterized cell, named by fixture, **not a bound and not the
  run's own error** (§3.6).
- 31 → 32 (2026-09-16, the UHF blocking disclosure's record half): the record
  gained the top-level **`symmetry_blocking`** block with its two channel
  members `alpha` and `beta` (`RunSymmetryBlocking`, §3.7), six keys per
  channel — `action`, `blocked_solve_count`, `plain_solve_count`,
  `generator_commutator_norms`, `max_generator_commutator_norm` and
  `tolerance`. Keys added, so a consumer must notice; no existing key moved,
  was removed, or changed nullability. It closes the same shape of gap 30 → 31
  closed: the blocking guard's demotion was visible only to a caller holding
  the in-process result, so a run that asked for blocking and silently walked
  the plain path read as though the irrep blocks had been used. Two rules a
  consumer keys on: the block is **absent** when nothing was requested — the
  guard enum's `kNotRequested` IS the absence, so blocking-was-used cannot be
  read off a run that never asked — and inside a present block a channel that
  disclosed nothing is **null**, never a fabricated `kUsed`. The `action` word
  is the deciding code's own (`kUsed` / `kDemoted` / `kRefused` /
  `kUnavailable`), never re-derived from the counts beside it.
- 32 → 33 (2026-09-16, the disk-tier ERI store's request surface): the
  **`[method] eri_cache_store`** input key (§2.1) and the `resources_resolved`
  **`eri_store`** block (`RunEriStore`, §3.6) added together. Keys added, so a
  consumer must notice; no existing key moved, was removed, or changed
  nullability. It closes a gap of a different shape from 30 → 31 and 31 → 32,
  and worth naming because it was invisible from the record side: the
  engine-decorator seam LANDED earlier (`FockBuildOptions::engineDecorator`,
  the disk tier, `integrals/include/qcx/integrals/eri_cache.hpp`) with
  NO input by which a run could ask for it, and a seam no input can reach is
  dead code however well it is tested. Three rules a consumer keys on. The
  first is the spelling: a path IS the request, the `[scf] checkpoint_file` and
  `[properties] molden` shape, with no separate boolean to disagree with it,
  and absent or empty is ONE not-requested state that leaves every run
  bit-identical to before the key existed. The second is the disclosure: the
  block is present on the honoured AND the demoted arm, so `demoted` beside the
  untouched `path` is what tells a reader the store was requested and did not
  run, while `engaged` names what actually ran — a demotion a record could not
  show would be the forbidden substitution wearing the demotion's name. The
  third is that the counters are the DECORATOR's, mirrored from
  `CachedEriStats` after the SCF (the driver owns the decorator and is the only
  module that may link both sides), never recomputed in the record, and
  they read 0 when the store did not engage rather than standing in for the
  `engaged` word. **The driver half is a follow-on**: this change lands the
  schema's two halves — the key and the block — and the call site that reads
  the key and installs the factory at the builder's Create is `driver/`'s, so
  at this revision a run that names the key still runs today's in-memory tier.
  That is the half-applied state and it is recorded here rather than left to
  be discovered, because a document that described the mechanism as wired would
  be the precise disagreement — prose against the landed code — this version
  history exists to prevent.
- 33 → 34 (2026-09-17, the `[grid]` input block and the record's physics, one
  bump for two rows — the input half and the record half together):
  keys added on BOTH sides, so a consumer must notice; no existing key moved,
  was removed, or changed nullability. **Input**: the **`[grid]` block** (§2.1),
  six optional keys — `radial_points` 75, `angular_points` 302, `alpha` 0.5,
  `radial_exponent` 2, `trim_weight` 1e-15, `block_target` 1024 — each
  defaulting to the engine's own compile-time value, so **an absent block
  builds exactly the grid every run built before the block existed**: the
  defaults state current behaviour rather than moving it. Each value is refused
  **BY NAME** rather than clamped — `>= 1` on `radial_points`, `radial_exponent`
  and `block_target`, a finite `alpha > 0` and a finite `trim_weight >= 0`
  (both have a silent nonsense in reach: `alpha = 0` collapses every radial
  point onto the nucleus, a non-finite `trim_weight` drops every point), and
  `angular_points` against the shipped Lebedev sizes, whose refusal lists them.
  The block is **refused on an `rhf`/`uhf` run**, which integrates no density
  functional and so builds no XC grid — the `XcKeyPolicy` rule
  `method.functional` and `method.screening_tolerance` already follow.
  **Record**: three top-level members — **`method`** (always present; the
  input's own four words, so a consumer reads back the word it can write into a
  file), **`functional`** (the registry's canonical name, not the file string
  re-read) and **`xc_grid`** (the six resolved settings) — with one presence
  rule doing real work: `functional` and `xc_grid` are written **exactly on a
  Kohn-Sham run and ABSENT on a Hartree-Fock one, never null and never a block
  of defaults** standing in for a grid that never existed (§3.1's null-honesty
  rule). That absence is the statement, and it is why the grid could not have
  been disclosed as ignored on a record that had no grid to ignore. A consumer
  keying on `method` is reading physics the record could not name at all before
  this bump: it named the Fock builder (`resources_resolved.selection`) and no
  method, no functional and no grid. **The generator-side consequence, recorded
  rather than left to be discovered:** `io/` gains a PRIVATE,
  implementation-only dependency on `qcx-grid` so the `angular_points` refusal
  can list the shipped sizes instead of copying them, which is a module
  edge a reader of the DAG should see here rather than find in a build file.

- 34 → 35 (2026-09-17, the builder-selection axes; the owner's ruling of the
  same day): **keys added on both sides**, so a consumer must notice; no
  existing key moved, was removed, or changed nullability. **Input**: the
  **`[builder]` block** (§8) — three orthogonal keys, `integral_family`,
  `storage_tier` and `execution_backend` — which replace the one conflated
  `[method] fock_builder` key. That key is **DEPRECATED and still ACCEPTED**:
  it is never refused, and every input file ever written resolves exactly as it
  did. **Record**: one top-level block, **`builder_axes`** (§8.4), carrying the
  RESOLVED family, tier and backend, the `requested_by` word that says which of
  the three spellings produced the selection, and the deprecated spelling
  itself with a `deprecated_key` naming it when the old key was used. The bump's honest
  reason is §8.1's: the old key conflated two axes that answer different
  questions, and the record carried the family word in a vocabulary where a
  lean run reads `"direct"`, so a reader could not see the TIER at all without
  already knowing which words were tier words.

- 35 → 36 (2026-09-17, the predictive memory model's REMOVAL; the owner's
  ruling of the same day): **a key REMOVED**, so a consumer must notice. The
  `resources_resolved.memory_model` block and every key under it are gone,
  with `driver/src/memory_model.cpp`, its header and its test — the driver's
  hand-derived prediction of a run's commit peak, deleted rather than
  repaired because a hand-derived formula drifts from the allocator by
  construction and this one had been caught wrong by 5×, 15× and 81 percent
  across its terms, plus an extrapolation from a fitted span to nine times
  beyond it. **No enforcement was removed**: the job-object cap
  (`memory_cap_gib` / `in_process_cap_applied` / `cap_note`) is untouched,
  and the rung a run takes is still decided by the integrals engine's own
  Create-time estimates (`mode_record`). Two consequences a consumer must
  know: `workspace_budget.capacity_bytes` is now the cap itself rather than
  cap-minus-modeled-base (the engine's exact estimate decides the rung
  against the whole cap), and the one fact the removed block carried that no
  other key carries — the direct family's within-family member — is read
  from `selection.builder_member` (schema 15), which names the member rather
  than flagging it.

- 36 → 37 (2026-09-20, the far field's accuracy budget reaches the record):
  **keys ADDED** to `qfmm_model` (§3.10), so a consumer must notice. The
  block already carried the model the composed-QFMM builder's Coulomb half
  ran; it now also carries the accuracy that model was held to and the
  a-priori truncation bound's verdict on it — `error_aware_admission`, the
  arm's own flag; `far_field_budget`, the preset budget the field was held
  to; `per_interaction_budget`, the share each far interaction was allowed;
  `geometric_far_pair_count`, the count that share was split over;
  `pairs_moved_to_near_field`, the pairs the budget moved into the near
  field; `fell_through_to_cap`, the interactions that ran at the order cap
  with their bound above that share; and `worst_truncation_bound`, the worst
  such bound. **Nothing about the far field changed** — the engine computed
  every one of these already and the builder's own record carried them — and
  the document is the only thing that moves: a run whose far field missed the
  budget it was held to was previously indistinguishable from one that met
  it, because the miss was recorded in the C++ record and dropped at the
  serializer. Two consequences a consumer must know: a nonzero
  `fell_through_to_cap` is a statement that the a-priori bound does **not**
  certify that run's far field, and it is not the measured error (the bound
  is a worst-case-over-distributions bound, so a nonzero count beside an
  inside-budget energy is expected and is the honest record of the two
  different questions); and the seven keys are written on every present
  block, with `error_aware_admission` as the answer for a build that did not
  arm the error arm, rather than omitted, so absence never means "the budget
  was met".

## 8. The builder-selection axes (schema 35)

### 8.1 What was wrong, and what the ruling is

`[method] fock_builder` carried words from **two axes that answer different
questions**. A **family** answers *where the integrals come from*; a **tier**
answers *where that family's working set lives*. Two tier words rode that one
key — the direct family's `lean`, and `in_memory` as an alias of `direct` — and
two more tiers were reachable only as rungs inside a family's own builder
(`blocked`, `disk`). The owner's ruling of 2026-09-17 splits the axes, keeps
every old spelling as a **compatibility alias**, and deprecates the key. It is
*normalize fully, deprecate the old key*, not *break callers*: nothing that used
to parse is refused, and nothing that used to run changes behaviour.

The leak is sharpest on `gpu`, and the fix is not a judgement call — it is what
the builder's own header says. `GpuJkFockBuilder` does "the same preparations as
`DirectJkFockBuilder::Create` plus the device side", with "the CPU builder's
screening semantics" and "its two-pass fp64 + certified-fp32 structure". A GPU
builder whose algorithm is the direct family's is a **backend**, not a family.

### 8.2 The axis table

One axis, one question, and every current spelling mapped to the combination it
selects.

| axis | values | requested by |
|---|---|---|
| `integral_family` | `direct` \| `ri_j_link` \| `ri_jk` \| `qfmm` | `[builder] integral_family` |
| `storage_tier` | `lean` \| `in_memory` \| `blocked` \| `disk` | `[builder] storage_tier` (the first two only — §8.3) |
| `execution_backend` | `cpu` \| `gpu` \| `gpu_split` | `[builder] execution_backend` |

The legacy spelling of each combination, which is what a file written before
schema 35 says:

| deprecated word | `integral_family` | `storage_tier` | `execution_backend` |
|---|---|---|---|
| *(absent)* | `direct` | `lean` at `nBasis <= 1000`, else `in_memory` | `cpu` |
| `direct` | `direct` | `in_memory` | `cpu` |
| `in_memory` | `direct` | `in_memory` | `cpu` |
| `lean` | `direct` | `lean` | `cpu` |
| `ri_j_link` | `ri_j_link` | `in_memory` | `cpu` |
| `ri_jk` | `ri_jk` | `in_memory` | `cpu` |
| `qfmm` | `qfmm` | `in_memory` | `cpu` |
| `gpu` | `direct` | `in_memory` | `gpu` |

Two cancellations are deliberate and each is the existing rule, not a new one.
`in_memory` is an **alias** of `direct` — one `BuilderKind` value comes out of
the two spellings, so no record and no refusal ever has to decide between them
(the same rule the deprecated key already followed). `lean` fills **no family
slot**: it leaves `method.builder` empty and sets `leanDirect`, and its axis
spelling does exactly the same, so the two vocabularies resolve to one value.

### 8.3 Where each tier value is spelled, and why not all four are one key's

`lean` and `in_memory` are the direct family's within-family choice, which is
why the deprecated key carried them beside its family words — they are
`[builder] storage_tier`'s. The axis's other two values are **named on the axis
and refused at that key**, each with the reason it is not this key's:

- **`disk`** is already an axis word with a home: `[method] ri_tensor_mode`
  (the rung selector), whose **force** lives at `[diagnostics] force_disk_ri`
  because the key-split rule of 2026-09-13 put it there. A `storage_tier = "disk"`
  spelling would undo that ruling and create a second spelling of one mechanism,
  so it is refused **by name with the owning key as the remedy** — the same
  posture schema 22 gave the retired `forced_disk` word.
- **`blocked`** is integrals' `RiFullFockRung::kBlocked`, the full-RI transform
  accumulated in auxiliary shell ranges. It has **no request surface at all**;
  it is named on the axis so the record can report it.

`gpu_split` is likewise named on `execution_backend` and refused as a request:
the intra-build batch-partition candidate has no builder to execute, so
the axis carries its name while execution stays on the wired backends (the
`kGpuSplit` precedent, preserved whole). A programmatic caller that sets the
kind directly is still refused in `ValidateCombination`, which is where that
refusal has always been.

### 8.4 The record: `builder_axes`

Always present, at the top level beside `method` and `functional`:

| member | type | meaning |
|---|---|---|
| `integral_family` | string | `direct` \| `ri_j_link` \| `ri_jk` \| `qfmm` — what RAN |
| `storage_tier` | string | `lean` \| `in_memory` — the tier the SELECTION resolved |
| `execution_backend` | string | `cpu` \| `gpu` — what RAN |
| `requested_by` | string | `axes` \| `fock_builder` \| `ladder` — which spelling produced it |
| `legacy_spelling` | string | the deprecated key's own word; **absent** unless that key was used |
| `deprecated_key` | string | `method.fock_builder`; present exactly with `legacy_spelling` |

**Every word is read off the RESOLUTION, not off the input**, so the block
cannot name an axis the wiring did not take. `requested_by` is what keeps
`legacy_spelling`'s absence unambiguous: `"axes"` is the `[builder]` keys,
`"fock_builder"` is the deprecated key, and `"ladder"` is no builder key at all
(§2.1's size ladder). The selection record
(`resources_resolved.selection`) is unchanged and keeps the legacy word and the
within-family member beside this block.

**The tier member reports the SELECTION's tier and nothing else.** The two tiers
a rung key reaches are not restated here, because one fact has one home: the
disk tier is `resources_resolved.ri_tensor_mode`'s answer (its ladder decides at
the engine's Create-time estimate, *after* this selection is made) and the
blocked tier is the full-RI family's own rung. A reader asking "did this run
land on disk?" reads the rung block; a reader asking "what did the selection
resolve?" reads this one. The block therefore does not restate a rung decision,
and it does not claim one either.

### 8.5 The deprecation behaviour, stated at the one place a caller looks

- **What a caller sees when they use the old key: ACCEPTED, never warned, never
  refused.** Every word `[method] fock_builder` accepted before schema 35 is
  still accepted and resolves to the same run. There is no stderr warning,
  deliberately: every input file written before this change uses that key, so a
  warning would be noise on every existing file and every existing test — and
  the repo's own precedent for an accepted alias (`in_memory`) warns nowhere
  either. The disclosure surface is the RECORD, not the terminal.
- **Where the acceptance is recorded**: `builder_axes.legacy_spelling` (the
  word) and `builder_axes.deprecated_key` (the note), both present exactly when the
  old key was used, and `requested_by = "fock_builder"`. A reader holding the
  JSON can therefore tell which vocabulary the input was written in without the
  input file.
- **What IS refused**: writing both spellings in one file. The axes and the
  deprecated key are two spellings of one request, so the pair is refused **by
  name**, with both sides and the deprecation named in the message — the
  key-split rule's *one request, one key*. An agreement check was rejected as
  the alternative because it would have to elect one spelling as authoritative
  and silently leave the other unread.
- **Unknown words** on any axis are refused by name with the accepted list, the
  same posture the deprecated key's refusal already had.
