# Sonar finding review

This review covers the 53 unresolved project findings returned by SonarCloud on
2026-09-12 while the corrective branch starts from `3eb221b`. It is not a claim
that a new hosted analysis has resolved them. Exact-head CI and Sonar analysis
must be checked after publication. Scientific YAML and model artifacts are not
remediation targets.

## Complete finding inventory

The API groups the findings as follows. Counts refer to findings, not unique
defects; two bootstrap warnings point to the same branch.

| Rule | Count | Disposition |
|---|---:|---|
| `githubactions:S8544` | 2 | Fresh CI installs use the frozen dependency lock |
| `text:S8565` | 1 | Add `uv.lock`; do not sync the qualified historical environment |
| `shelldre:S7688` | 3 | Use Bash `[[` with unchanged conditions |
| `python:S5863` | 2 | Replace self-comparisons with independently constructed expectations |
| `python:S6982` | 3 | Explicitly place returned/restored models in evaluation mode |
| `python:S6985` | 1 | Restrict trusted resume deserialization and checksum the same input bytes |
| `pythonsecurity:S8705` | 1 | Construct a fixed child operation from parsed arguments without a shell |
| `pythonsecurity:S2083` | 2 | Validate positive integer PID; retain explicit caller-selected manifest destinations |
| `pythonsecurity:S8707` | 11 | Reviewed explicit local CLI/configuration paths; preserve runtime relocation |
| `python:S1244` | 25 | Exact frozen identities, discrete target parameters, and binary diagnostics; no tolerance |
| `pythonbugs:S2583` | 2 | Empty finite bootstrap population is reachable; add both-branch regressions |

## Findings that must not change scientific behavior

Exact equality in paper configuration, evaluator inventory, representation
schemas, diagnostics, and continuation compatibility is intentional. These
values identify a frozen experiment or select a discrete supported distribution;
they are not estimated measurements. A tolerance could authorize a different
experiment. TCA compares the complete frozen parameter mapping and its tests
reject one-ULP changes to each numeric execution parameter.

The public CLI accepts operator-selected configuration, input, output, and
runtime-root paths. Absolute paths outside the repository are supported, not
evidence of traversal by themselves. The reviewed `config.py` sink opens the
explicit configuration path. Manifest I/O is a low-level API whose caller owns
the chosen destination. Confining these paths to the checkout would break the
documented relocated evaluator. Artifact-member paths remain separately subject
to containment and checksum checks. Unique temporary publication also prevents
clobbering an unrelated fixed-name temporary file.

The bootstrap empty branch is taken for an empty sequence or after removing all
non-finite values. A nonempty constant population reaches resampling and returns
constant bounds. Tests cover both cases without changing the estimator. The
pre-existing ordering of invalid-option checks on empty input is unrelated to
the actual frozen evaluation configuration and is not changed here.

## Evidence and limits

Focused regression sources are `test_probe_cache.py`, `test_paper_representations.py`,
`test_paper_sequences.py`, `test_runpod_deployment.py`, `test_paper_manifests.py`,
`test_tca_population.py`, and `test_reporting.py`. Existing CLI workflow tests
exercise configuration and data roots outside the repository. ADR 0030 records
the operational decisions.

No warning is suppressed through `NOSONAR`, a rule exclusion, relaxed numerical
tolerance, or disabled test. Reviewed false positives are distinguished from
code corrections; a clean local test run is not a hosted Sonar resolution.

## PR analysis follow-up

The initial PR #8 analysis added five `githubactions:S8541` findings for the
verification commands and one `pythonsecurity:S8705` finding for the benchmark's
baseline revision. Verification commands now explicitly use both `--no-sync`
and `--no-build`: dependency installation is confined to the preceding frozen
sync step. The benchmark accepts only a full lowercase commit SHA and places
it after Git's end-of-options marker. Its default is an immutable baseline,
not `HEAD`. Tests reject option-like and mutable revision arguments before
subprocess execution. A new hosted analysis must verify these corrections.

The next analysis confirmed zero new issues and hotspots, but reported 6.8%
new-code duplication. Synthetic learned/EWMA ledgers now have one shared fixture
builder. Test references retain independent legacy operations and exact-output
assertions without duplicating metadata construction. The rejected projected
matcher prototype was removed rather than maintained as a second statistics
implementation. These changes do not exclude files from analysis or relax the
3% gate; the next exact-head hosted analysis must confirm the measured density.
