# Optimized evaluation implementation

Result shard merging promotes each timestamp field to the finest input storage
unit, including typed empty shards. Timezones must match exactly, batch casts are
safe (overflow fails), and other type conflicts remain errors. No timestamp is
rounded or converted to another timezone. Input hashes, canonical row ordering,
atomic publication, and resume verification remain mandatory; see ADR 0024.

This document records software behavior for the downstream evaluator replacement.
It does not amend model parameters, TEST inclusion, or statistical estimands.

## Documentary compatibility

The immutable design freeze continues to bind the original scientific YAML and
specification. `implementation-document-amendment.json` records only the four
reviewed implementation/navigation document edits. The loader verifies each current
document hash, reverses the exact recorded edits, and requires the reconstructed
text to match the original frozen hash. The receipt also binds the unchanged design
freeze bytes. Scientific YAML and `PAPER_DESIGN.md` are ineligible for this mechanism.
Unrecorded edits and changed original text fail closed. The scientific config hash
does not include this implementation receipt and remains unchanged.

## Compact base artifacts

`evaluation_artifacts.evaluation_base` is IMPLEMENTED. The orchestration constructs
one base per fold outside the eight-method loop. VALIDATION supports semantic tests;
TEST access still requires the caller's runtime authorization and parameter freeze.
TRAIN retains the separate training cache because it samples shape origins.

The cache identity binds parameter-freeze SHA-256, source commit/tree, paper config,
sequence-manifest SHA-256, partition, and liquidity metadata. Scale and shape files
store targets alongside features. Rows sort stably by instrument, date, as-of,
sample ID, and (for shape) target bucket. No scientific float downcast is applied.
PyArrow uses 65,536-row groups and memory-mapped reads. Manifest entries record the
exact schema, row count, and SHA-256 of each file.

Scale identities must be unique. Shape identities must refer to scale rows and
cover each available future bucket exactly once, with finite nonnegative shares
summing to one. Targets and features are sorted together. Embeddings attach using
the existing sample-ID join and preserve the frozen feature column order.

## Publication and reuse

`publish_frames` writes into a temporary sibling directory, syncs each data file
and the manifest, then renames the completed directory. Workers must own disjoint
destinations. A missing manifest, changed hash, schema, row count, or identity fails
closed. Partial temporary directories are not completion markers. Derived caches
are disposable; authoritative upstream files are never overwritten.

## Bounded stage merge

Forecast and TCA orchestration retain paths, not result DataFrames, after each
shard completes. `merge_result_shards` verifies the expected inventory against
the shard manifest hashes, sorts one shard at a time into temporary Parquet,
then streams disjoint canonical key ranges in fixed-size batches. It rejects
duplicate row identities and overlapping shard ranges rather than deduplicating
or silently mixing populations. Null shared seeds retain their Parquet null
semantics when combined with integer model seeds. No scientific values are downcast.

The completed file is atomically replaced before its completion manifest is
published. The manifest binds source shard hashes, sort keys, execution identity,
schema, row count, and output hash. Resume verifies that binding and reuses the
completed file without rewriting it. A file without its completion manifest is
not a completed stage. Tests require identical merged bytes across input inventory
orders and equality with the original direct Parquet result, plus failure on
missing/corrupt inputs, duplicate rows, and overlapping ranges.

## Vectorized case metrics

`forecast_metric_frame` is IMPLEMENTED in the learned forecast loop. It outer-joins
actual and predicted shape rows and rejects
population mismatches before calculating cumulative-share distances ordered by
case and target bucket. Mean absolute cumulative-share differences are joined back
to scale sample identity. Log-volume error is computed over aligned arrays.

Learned predictions publish one atomic directory per fold/method/seed, containing
scale predictions, long-form conditional shares, and case metrics. Its manifest
binds the base manifest, model manifest, embedding bytes, source commit/tree,
parameter freeze, and scientific config. Re-entry verifies the files and skips
already-published learned predictions. The synthetic orchestration test requires
one base build and two prediction calls across two complete invocations for the
two shared methods. No historical run has qualified this implementation yet.

## Remaining execution work

`evaluation_workers` is IMPLEMENTED in forecast orchestration. A bounded scan of
the raw corpus writes checksum-bound per-instrument profile files once for reuse
across folds. Each EWMA process receives only paths and identities, reads one
instrument's base and profile, and publishes scale, shape, metrics, and minute
forecast ledgers atomically. `EXECSIM_EVALUATION_WORKERS` defaults to 16; spawned
workers inherit one OpenMP/MKL/OpenBLAS/NumExpr thread. The parent environment is
restored after the pool. Task output paths must be unique. Re-entry verifies prior
shards and does not republish completed ones. No large frames cross process pipes.

EWMA minute evidence (schema `paper-ewma-ledger-v4`) records the exact end token. Full-session forecast evaluation
uses end token 26; TCA uses end token 24. These are separate estimator requests:
truncating a full-session normalized historical profile does not generally equal
estimating the shorter requested window. The fixture explicitly demonstrates this
difference and requires exact agreement with the original TCA-window estimator.
The ledger records every intra-token minute requested by replay, including the
original normalized shares, expected total, warnings, and cutoff. It does not
substitute truncation for a new EWMA window. The fixture requires exact public
forecast equality for all 15 offsets and rejects unavailable or different windows.

### Unavailable history

`HistoricalForecastUnavailable` is a `ValueError` subtype for no prior sessions,
no complete prior requested windows, or no positive eligible historical volume.
Ordinary core callers still receive an exception. Paper workers catch only this
type and atomically publish `unavailable.parquet` with `fold_id`, `sample_id`,
`instrument_id`, `symbol`, `session_date`, `as_of`, `end_token`, `generated_at`,
`status=EWMA_UNAVAILABLE`, and `reason`. Availability depends only on prior data.
Other errors remain fatal. No estimator, target, or minute observation is changed.

The scale ledger retains every requested identity, with an explicit status and
null prediction for unavailable full-session requests. Shape and metric ledgers
contain only available full-session cases. Minute ledgers independently preserve
available exact TCA requests; a missing full-session request cannot exclude a
valid shorter request. Empty availability and metric artifacts are valid typed
artifacts, not missing work. Resume verifies all five files and requires v4.

TCA discards any partial simulation when a required EWMA request is unavailable
and emits an identified unavailable row with no performance metrics. Other methods,
balanced sides, order quantities, and market eligibility remain unchanged.
Pairing excludes only explicitly unavailable rows and counts them as dropped;
non-finite available metrics fail. Zero-overlap comparisons do not run bootstrap.

Reporting retains forecast-unavailability, TCA-unavailability, zero-overlap, and
all-method coverage appendices. Confirmatory dense/sparse comparisons retain their
pair-specific sample, independent of EWMA availability. Descriptive all-method
tables use their declared common-case population. See ADR 0023 for rationale.

`forecast_ledger.PaperForecastLedgerProvider` is IMPLEMENTED in the learned TCA
provider factory. The factory neither reconstructs model feature frames nor calls
LightGBM. It validates the expected ledger identity and
checksums, selects one instrument/session, checks fold and cutoff identity, and
requires the exact future bucket grid for each requested as-of sample. It expands
stored token forecasts using the same TRAIN-only minute profile as the direct
provider. Between boundaries it truncates the prior minute forecast, preserving
within-token offsets. The fixture requires exact `VolumeForecast` equality at the
boundary, +1, +7, +14, and the next boundary, and rejects incompatible model/fold/seed
and missing as-of identities.

Each fold's learned ledger is SHA-verified once per process. A `VerifiedArtifact`
records file size, nanosecond modification/change timestamps, and inode after
verification, and rejects changed files or expected identities on reuse. This is
an immutable-file execution contract, not permission to mutate verified artifacts.
TRAIN-only within-token profiles are reused per instrument in that fold.

`tca_workers` is IMPLEMENTED in TCA orchestration. Tasks are independent trading
dates, preserving the full date population used by `balanced_sides`. Workers read
date bars, ADV, TRAIN profiles, and universe tables from verified Parquet inputs.
They run the existing deterministic replay for main and sensitivity sizes and
atomically publish both outputs. Positive runtime worker counts and disjoint output
paths are required. The default is 16 spawn workers with one native numeric thread.
Learned forecasts come exclusively from the earlier ledgers. The fixture compares
two date shards with serial replay and checks sides, cost output, and no-rewrite
resume behavior. EWMA within TCA now reads the exact minute ledger, without
rebuilding history or estimating forecasts. Each worker verifies immutable learned
and EWMA ledger hashes once, retaining only verification metadata between date
tasks. Session-level baseline providers are reused across order-size scenarios.

## Bounded TCA input preparation

TCA no longer loads the complete raw corpus. The shared compaction scanner projects
the required market columns for the union of the frozen main and sensitivity
instruments, preserving native numeric values. Its source inventory must equal
the corpus bound by the forecast ledgers. It writes per-instrument derived market
files without modifying upstream data.

`tca_inputs.prepare_tca_history` reads one instrument at a time. The original ADV20
and within-token profile functions execute in original corpus order before sorting
the derived replay rows. All fold cutoffs are bound to the history manifest.
Sorted bars use 390-row groups, allowing a date request to read only nearby row
groups. ADV, TRAIN-only fold profiles, and the session inventory are separately
persisted and checksum-verified on resume. The orchestrator assembles only one
date across the fixed instrument population, preserving balanced-side semantics,
publishes its inputs, and releases those frames before the next date.

A 35-session fixture requires exact equality of raw bars, ADV20, and profiles at
two cutoffs. It also executes the real preparation/orchestration with the former
full-corpus loader made unavailable, verifies each emitted date input, and checks
resume and changed-cutoff rejection. Replay-worker equivalence is tested separately.

Before any TCA worker is launched, the orchestrator resolves each selected
instrument/session through the existing `resolution-aware-v2` assessor and keeps
only `tca_window_exact` cases. An early close or an instrument-specific consumed-
window gap removes that case only; `balanced_sides` receives the surviving
date-level population, and dates with no surviving cases create no task. A bounded
preflight then verifies every required learned ledger, EWMA availability ledger,
fold/cutoff/sequence identity, exact as-of grid, and future-bucket grid. Missing or
incompatible derived evidence aborts before replay rather than silently changing
the scientific population. EWMA validation treats the artifact manifest as the
fold authority (the published scale rows intentionally have no `fold_id` column)
and scopes both `minute-forecasts.parquet` and `unavailable.parquet` to the
current session and `end_token == 24` before checking identities and coverage.
Preflight groups eligible dates by instrument and reuses one bounded read of each
ledger while validating that instrument, rather than rereading the same files for
every date. See the existing resolution-quality contract and
`docs/ADRs/0009-separate-data-quality-by-resolution.md`.

## Evaluator reseal and isolated output namespace

To supersede an existing resealed execution, first create the typed chain
receipt (for example, with `prepare-evaluation-supersession`) and then pass it
to `reseal-evaluation`:

```text
execsim ml paper prepare-evaluation-supersession \
  --superseded-execution artifacts/paper/protocols/sparse-jepa-v2/evaluation-executions/b7eee96 \
  --supersession-output artifacts/paper/protocols/sparse-jepa-v2/superseded/SECOND_RESEAL/supersession.json \
  --reason "corrected pre-evaluation provenance contract"
```

The command records a typed, checksum-bound predecessor; it does not copy or
reuse any predecessor result artifacts.

`execsim ml paper reseal-evaluation` accepts `--evaluation-root` and a required
`--supersession-receipt`. It requires locked-evaluation runtime approval and clean
committed source. The output root must be a new child of the authoritative
artifact root's `evaluation-executions` directory. A nonempty unsealed directory
is rejected. The original parameter freeze, TEST-ready/open receipts, and fitted
artifacts are never rewritten.

The new `execution.json` binds source commit/tree, scientific config, the existing
parameter-freeze hash, original evaluation source, supersession hash, cache/ledger
schemas, and the exact configured upstream matrix. It verifies both native
LightGBM boosters and grid-selection bytes per coordinate, JEPA safetensors and
compatibility files, checkpoint/export checksum links, all three embedding
partitions, and sequence manifests. Missing, extra, duplicated, or changed model
identities fail closed. The seal records zero initial completed stages.

Pass the same `--evaluation-root` to subsequent forecast, representation, TCA,
and report commands. Derived caches, numerical outputs, and report bundles are
directed into that namespace. Loaders accept the original parameter source only
through the verified new execution receipt. This is an implementation-source
bridge, not permission to regenerate parameters or reopen an unrelated TEST run.
Learned-ledger identities in both forecast and TCA must hash the base manifest
under this configured evaluation root, never a similarly named cache under the
upstream artifact root. A missing isolated base is an error even if a legacy
base exists. Regression tests exercise default and isolated forecast orchestration,
resume, and conflicting legacy caches for every learned-method family.
CLI help and synthetic receipt tests are executed; no historical seal has been
created yet. All original checkpoint bytes must be available on the execution
host before sealing can pass. `--representation-root` selects the immutable
checkpoint import used only by evaluation and its integrity receipts; it does not
redirect training. This permits complete CUDA originals to coexist with retained
partial/CPU-era copies without overwriting either. The selected import must stay
within the authoritative artifact tree so receipt paths remain portable.
Logical paths remain bound even when Windows directory junctions resolve the
embedding bytes into the original training worktree; the resolved bytes are
still SHA-verified. Each evaluator process hashes the complete upstream inventory
once, then checks file size, inode, and nanosecond modification/change times on
reuse. Changed files or source/config/root identity are rejected. Returned receipt
objects are copies, so callers cannot mutate the verified state.

Representation evaluation publishes an atomic coordinate directory for each
fold, geometry, and seed. Its identity binds the sequence, checkpoint,
compatibility, embedding manifest, parameter freeze, and evaluator source.
Completed coordinates are verified before any model is loaded or regime frames
are built; missing coordinates use the existing frozen capacity/probe procedure.
JEPA and LightGBM weights are never fitted by this stage. Date-level and summary
outputs are merged from the complete expected coordinate inventory using sorted,
checksummed result manifests. Corruption is an error, not a reason to silently
recalculate or replace a completed result. Synthetic orchestration tests prove
resume skips model loading and evaluation and rejects changed coordinate bytes.

Reporting verifies all six merged numerical input hashes and their evaluator
source/config identities before reading results. It builds the entire bundle,
including appendices, in a temporary sibling tree. A completion receipt binds
every output file's SHA-256 and all input hashes before atomic publication.
Resume checks the exact file inventory and bytes, not merely file existence;
an interruption leaves no authoritative partial report. Synthetic tests cover
interruption, unchanged reuse, changed source, corrupted output, and changed input.

Historical restart remains PLANNED. Existing unit checks
do not establish completion of these runtime stages. See ADR 0022 for rationale.

Before a resealed TEST execution can consume runtime data, forecast and TCA
stages resolve the universe and target corpus with `PaperRunConfig.data_path`.
The runtime universe must have the exact byte hash recorded by every fold's
sequence manifest, and all sequence manifests must agree on that identity.
Reseal also checks the complete configured primary JEPA final inventory: every
manifest must provide a non-empty training `code_commit`, and all coordinates
must share one commit.  Final-result-freeze retains the same check as defense
in depth.  For TCA, exact-window eligibility remains independent of ADV
availability, but every eligible case must have exactly one finite, positive
causal ADV20 row before worker launch; direct historical replay repeats this
check and raises rather than silently dropping a case.  These are fail-closed
identity and derived-evidence checks; they do not open TEST or change the
frozen estimand.

TCA ADV20 histories bind the runtime corporate-action manifest and use the
existing point-in-time split-factor convention to express every prior daily
volume in the target session's raw execution-share basis. The target volume is
excluded by the strict 20-session lag, and raw replay bars are not restated.
Action-bound histories use a new artifact identity, so a history built from
different corporate-action bytes cannot be reused.

First-generation reseals remain `paper-evaluation-execution-v2` for immutable
historical compatibility. A reseal that supersedes an existing resealed
execution uses a typed `paper-evaluation-supersession-v1` receipt and publishes
`paper-evaluation-execution-v3`. The v3 identity records both the original
TEST-open source and the immediate predecessor; validation checks the prior
`execution.json`, its checksum, and its immutable inventory before publishing
the new empty namespace.

Reseal cross-links the immutable inventory before publishing `execution.json`.
Each fold's sequence manifest must be the sequence named by every LightGBM
coordinate, JEPA checkpoint, compatibility record, and embedding export.  The
checkpoint universe identity and representation source commit must match the
parameter freeze; embedding normalization and paper configuration identities
must match the checkpoint; and hybrid LightGBM manifests must name the exact
TRAIN and VALIDATION embedding bytes that are present in the export.  A
complete but cross-coordinate-swapped artifact set is rejected before forecast
construction.

Learned inference materializes at most 2,048 scale samples and their complete
future shape rows per wide batch. A compact embedding partition is read once per
coordinate and attached by exact sample ID to each batch, rather than expanded
across the full fold's shape table. Scale order and complete conditional horizons
are preserved; noncanonical shape order or missing/duplicate embedding identities
fail closed. Tests compare full and batched raw, hybrid, and untrained-control
feature frames and actual native LightGBM predictions on synthetic inputs.
The untrained neural control retains its original 8,192-row encoder batches:
its compact 644-value vectors are computed once in original scale order, then
attached to bounded prediction batches. Changing the neural batch size produced
small CPU-kernel arithmetic differences on Linux, so the implementation preserves
that boundary instead of relaxing the equality check. Raw and embedding features
must match the unbatched path exactly in the regression fixture.

A bounded historical semantic check also passed on 47,990 observed ABT/SPY
minute rows from November 2023 through January 2024, exclusively Fold-1 TRAIN
and VALIDATION. All 144 requested forecasts matched the original prefix oracle
exactly across mean, median, previous, EWMA, pooled/unpooled scope, and three
requested minute windows. The local receipt records input SHA-256 values; this
was not a speed benchmark, model fit, or TEST effectiveness evaluation.

## Frozen input reuse (ADR 0029)

The representation evaluator can materialize the exact `_encoded_batch` output
once per coordinate and partition. Four contiguous files retain feature,
latent-target, observable-target, and complete-mask dtypes. A metadata stream
records the original batch boundaries, sample IDs, dates, and as-of positions.
Memory-mapped replay copies only the active batch to the execution device.
No incomplete rows are dropped during materialization. Statistics and probe
updates still perform their original mask selection and reduction order.

Cache manifests bind the evaluator coordinate identity, checkpoint and sequence,
device, PyTorch version, partition, and batching. All data files are checksummed;
missing checksums, changed bytes, unaligned shapes, or different identities fail
closed. Publication is atomic. Cache materialization restores the loader's RNG
state; replay consumes the same DataLoader iterator seed transition. Completed
scientific coordinate artifacts retain their existing independent publication
and resume contract. Disposable cache files are retired after publication so
storage does not grow with the full coordinate matrix.

The representation-stage thread policy accounts for affinity and cgroup v1/v2
quota, limits native pools, and sets inter-op parallelism to one. This is local
execution configuration, not a change to batch size or any scientific setting.

Forecast raw features are constructed in consecutive session groups, preserving
sample order, values, categorical inputs, and target/weight calculations. Existing
manifest-bound index caches replace repeated tiny-index reads. TCA workers read
one verified learned date slice and construct independent provider states for
each simulation. Identity/file-state checks remain active; old evaluation result
namespaces are never imported as completed outputs in a new execution.

These changes require real bounded benchmark evidence and exact fixture
equivalence. They do not imply empirical effectiveness or authorize evaluation.

TCA date inputs are published during the first population pass, after exact-window
and positive unique ADV20 validation. Dataframes remain bounded to one date.
All fold ledger preflights must pass before any worker launches. A failed
preflight may leave valid atomic input artifacts, never completed TCA results;
the same source-bound execution verifies those inputs on retry. This eliminates
the second history read and repeated window assessment without changing the
surviving cases or their order.

### Bounded probe benchmark

On the qualified Linux/CUDA host, run the reference and cached paths with the
same source, inputs, geometry, seed, row count, and native thread limit:

```bash
PYTHONPATH="$SOURCE/src" "$PYTHON" "$SOURCE/scripts/benchmark_frozen_probes.py" \
  --source "$SOURCE" --artifact-root "$ARTIFACT_ROOT" \
  --work "$BENCHMARK_ROOT" --mode reference --geometry sparse --rows 8192 --threads 4
PYTHONPATH="$SOURCE/src" "$PYTHON" "$SOURCE/scripts/benchmark_frozen_probes.py" \
  --source "$SOURCE" --artifact-root "$ARTIFACT_ROOT" \
  --work "$BENCHMARK_ROOT" --mode cached --geometry sparse --rows 8192 --threads 4
```

The script reads TRAIN/VALIDATION only, runs the unchanged 20-epoch probe ladder,
and uses a second VALIDATION loader in place of TEST for timing/scoring. It emits
only timings, resource measurements, and a digest of mathematical outputs.
Source bytes bind benchmark cache reuse. The reported evaluator wall time
includes cache construction but excludes initial checkpoint/index loading.
These are operational qualification outputs, not paper results.

The execution-verification memoization guard includes every receipt and upstream
file reached during recursive supersession verification, not only the immediate
predecessor. After an initial successful verification, changing an older ancestor
execution or supersession receipt still fails closed. This enforces the existing
ADR 0028 chain contract for third and later generations without altering schemas
or copying old results.

### Performance qualification evidence

On the retained RTX 3090 host (27.2 effective cgroup CPUs), the isolated
TRAIN/VALIDATION benchmark at source `44a12b2` used 8,192 sparse TRAIN samples,
512 VALIDATION samples, a second 512-row VALIDATION scoring loader, batch size
256, and the unchanged 20-epoch probe ladder. With four native threads, reference
evaluation took 131.164 seconds and cached evaluation, including materialization,
took 26.775 seconds (4.90 times faster). CPU time fell from 187.939 to 26.619
seconds. Peak RSS was 2,919,992 versus 2,962,944 KiB. Both mathematical-output
digests were exactly
`dc1a0e698ae6cd229088f32224ac8abb6bc9c7c0c7c60ce8cdea7bd9c7633d26`.
These evaluator timings exclude initial checkpoint and index loading and do not
predict a full-coordinate completion time. Total OS threads were 84, including
idle native/CUDA pools; four is the computational pool limit, not a claim that
the process contains only four threads.

A subsequent cached-input comparison at one, two, four, and eight native threads
preserved all nonnumeric identities and both selected ridge alphas. The maximum
absolute mathematical-output difference was 1.7764e-15; comparisons used
`rtol=1e-10, atol=1e-12`, excluding telemetry. No TEST samples were consumed.
Warm-cache timings are not substituted for the cold-materialization comparison
above. The four-thread default remains the fully paired reference/cached policy.

Other bounded input-only checks found exact frame equality for session-batched
forecast preparation (42.45 to 24.86 seconds for 128 sessions), verified index
reuse (6,136 reads to one for 134,992 rows), and shared learned ledger date reads
(60 reads to two). Single-pass TCA input preparation on ten TRAIN dates preserved
the exact inputs while reducing reads from 38 to 20. These checks did not score
historical model effectiveness.

An OSQP workspace-reuse prototype was rejected: it failed integer-capacity
projection on a bounded TRAIN replay, while the unchanged solver passed. Solver
setup represented less than one percent of the measured replay profile, so no
production solver change was retained. Result-frame construction was also less
than one percent; a summary-only simulator was not introduced. A synthetic
10,000-replicate block-bootstrap timing was under two seconds and did not justify
changing the report estimator. These observations are bounded profiling evidence,
not a full historical TCA or reporting performance claim.

### Operational artifact safeguards

The coordinate-result schema and encoded-cache schema are distinct. Construct
the complete cache identity with `encoded_probe_identity` for both creation and
retirement, including batching, partition, device, and PyTorch version. Retirement
verifies all partition identities and file checksums before deleting any cache
file; it never ignores the schema to permit cleanup. The published coordinate
remains the scientific result and must exist before retirement.

Trusted resume loading verifies the exact bytes passed to PyTorch's restricted
weights-only loader. Only the existing NumPy MT19937 reconstruction types are
additionally allowed, within a scoped context. Unsupported object types fail;
existing safetensors and continuation files are not rewritten. JSON publication
uses unique same-directory temporary files and atomic replacement, preserving
the prior receipt if replacement fails. See ADR 0030.

Learned TCA ledger preflight builds date and case-position indexes once per
instrument. It passes physically ordered date slices through the existing
identity, cutoff, origin, duplicate, future-bucket, and share checks. Indexing is
operational: it does not remove an eligible case or replace preflight with trust
in a cache. A missing eligible date still fails before workers launch.

The [Sonar review](SONAR_REMEDIATION.md) accounts for the full project finding
inventory and distinguishes operational corrections from intentional exact
identity and runtime-path contracts. It does not replace exact-head hosted gates.

### Indexed preflight benchmark

Run the bounded synthetic comparison from the repository root:

```powershell
.venv/Scripts/python.exe scripts/benchmark_tca_preflight.py --baseline-revision 3eb221bf787421fd4d5397b725397861b403fdaa --dates 128 --origins 22 --repeats 2
```

The benchmark uses eight learned ledgers and one EWMA ledger for one synthetic
instrument, preserves all preflight checks, and extracts the old validator from
the specified Git revision. It never reads historical TEST data or calls a
forecast provider. On the local Python 3.13 environment, two paired runs gave
median elapsed times of 17.119 seconds before indexing and 11.968 seconds after
indexing (1.43 times faster). Learned validator time was 13.245 versus 7.247
seconds. Both paths made 19 Parquet reads and returned 326,656 rows.

The benchmark imposes no CI timing threshold. These results quantify only
preflight work on the declared fixture, not historical replay, report matching,
or end-to-end completion time. Exact-output and adversarial regressions remain
separate gates.

### Guarded forecast fast paths

Contiguous shape groups retain the original per-group NumPy max, exponential,
sum, and assignment order while avoiding pandas group-index and row-write
overhead. Interleaved, categorical, or missing keys retain the original path.
Metric inputs bypass merge/sort only when unique typed keys align exactly and
future buckets ascend within contiguous cases. Fallback share validation stays
before sorting. Tests include near-tolerance sums and malformed populations.

Learned forecast providers retain private immutable metadata for the minute grid
they construct. Truncation slices only if the cached timestamp tuple is that
grid object and the entire requested tuple matches a contiguous slice. Generic
requests, including unsorted or duplicate timestamps, retain dictionary semantics.
The normalization reduction and returned forecast fields are unchanged. See ADR
0031.

The synthetic helper benchmarks in `test_forecast_fast_paths.py` assert exact
outputs before timing. For 10,000 cases and 80,000 shape rows, stub-model
`predict_frames` took 1.237 seconds on the reference path and 0.069 seconds on
the guarded path. Full metric construction took 0.078 versus 0.063 seconds.
These timings exclude real LightGBM prediction cost. The 300-minute truncation
helper took 0.334 versus 0.194 seconds; the full synthetic simulation parity test
also passes. No historical effectiveness was inspected for these comparisons.

### Other preparation costs

Run the bounded preparation audit from the repository root:

```powershell
.venv/Scripts/python.exe scripts/benchmark_evaluation_preparation.py --ewma-samples 128 --sequence-samples 100000 --report-cases 25000 --repeats 3
```

On the local environment, 2,048 exact EWMA requests took a median 0.994 seconds.
Each offset retains its independently eligible historical window; substituting
truncation would not preserve the estimator. Fixed-grid construction for 100,000
samples took 1.427 seconds, with 0.094 seconds in grouping/sorting. Reusing
DataLoader instances is not adopted because it can change generator state.

For 97,877 synthetic method rows and three complete-case comparisons, narrowing
columns before copying took 0.181 seconds versus 0.171 seconds for the existing
matcher. That prototype is not retained. Peak RSS was unavailable in this local
benchmark; it is not reported as zero. These bounded observations do not rule
out larger-corpus bottlenecks or replace the required full-coordinate runtime
observation. Cross-process checksum verification remains unchanged.
