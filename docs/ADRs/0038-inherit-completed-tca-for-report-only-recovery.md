# ADR 0038: Inherit completed TCA for report-only recovery

Status: accepted

## Context

The d0f8877 execution completed all 250 TCA date shards and both merged outputs.
Reporting then failed while pandas 3.0.5 serialized a 33,930-by-23 sensitivity
table to LaTeX. The Styler element limit truncated the internal rendering data;
the strict row pairing raised an exception. The numerical results were not the
failure. Replaying TCA would discard valid work without correcting serialization.

## Decision

Use a scoped pandas option context that raises `styler.render.max_elements`
above each complete table's cell count. Restore the caller's option on exit,
including errors. Export every row and retain all existing formats and values.
Do not downgrade pandas, suppress export failures, or truncate paper tables.
See the [pandas options contract](https://pandas.pydata.org/docs/user_guide/options.html).

Introduce inheritance v3 and execution v5 for the explicitly authorized
report-only recovery. The only inherited stages are forecast, representation,
and TCA; the invalidation frontier is report. Keep the immediate predecessor
separate from each stage's original producer and execution receipt. Bind the
complete expected TCA date-shard inventory, merge receipts, merged Parquet files,
and aggregate manifest by checksum. Verify native identities and exact inventory
membership, not just file existence or an asserted completed count.
The report-only writer requires the independently audited fold/date inventory
as an explicit recovery input. Persist that exact set in v3 and compare it with
both merge receipts, input dates, and physical shard directories on every
verification. Do not infer TCA eligibility from sequence dates or silently
derive the expected set from the potentially incomplete result directory.

Resolve inherited stages at their original paths. New report and final-freeze
outputs name their own evaluator and retain all three upstream producers. The
explicit report-only supervisor may dispatch only report and final-result-freeze
after native verification. Inheritance v1/v2 and execution v4 keep their existing
TCA-restart semantics; no old receipt is rewritten or migrated.

## Alternatives considered

Replaying TCA is unnecessary under the user's report-only authorization. Copying
or relabeling results would misstate their producer. Extending v2 in place would
silently change a durable contract. A generic stage DAG or arbitrary resume plan
would broaden the authorized recovery unnecessarily.

## Consequences

This extends ADRs 0035 and 0037 only for completed, verified TCA inheritance.
Scientific configuration, models, populations, estimators, and numerical results
are unchanged. Missing shards, altered merges, incompatible sources, and changed
ancestors fail closed before report dispatch. Qualification requires the actual
production-sized shadow report, complete serialization, mixed-source final
freeze, and repeat-stable receipt checks before the official restart.
