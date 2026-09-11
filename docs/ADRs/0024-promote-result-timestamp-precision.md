# ADR 0024: Preserve timestamp precision when merging results

- Status: Accepted
- Date: 2026-09-11
- Complements: [ADR 0023](0023-record-unavailable-historical-forecasts.md)

## Context and evidence

The a70b39e forecast execution produced all expected fold shards, then failed
while merging availability records. Arrow rejected generated_at timestamp fields
with identical America/New_York timezones but different millisecond and
microsecond storage units. This is a storage contract defect, not a model result.

## Decision and alternatives

For each top-level timestamp field, select the finest unit present across all
input schemas, including empty shards. Preserve timezone identity exactly and
cast batches with Arrow's safe conversion. Reject timezone differences,
timestamp/non-timestamp conflicts, overflow, and unrelated incompatible types.
Keep strict schema unification for other fields. Do not enable general permissive
promotion or downcast timestamps: either would broaden the contract unnecessarily.

## Scientific consequences

Instants, nulls, row identities, values, ordering, and the estimand are unchanged.
No metrics inform the fix. Input bytes and their source manifests remain immutable;
the merged artifact records its promoted schema and authoritative input hashes.
The successor evaluator receives a new source seal. Old artifacts are not relabeled.

## Verification

Synthetic tests cover mixed precision, fractional timestamps, nulls, empty shards,
order-independent bytes, resume, timezone/type rejection, and safe overflow failure.
