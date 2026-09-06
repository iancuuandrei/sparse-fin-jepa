# ADR 0011: Source split actions with conservative availability

- Status: Accepted
- Date: 2026-09-06
- Scope: paper data acquisition and point-in-time preprocessing

## Context

The frozen paper protocol requires point-in-time split treatment with stable instrument identity, but the historical orchestration had only an ingestion boundary for a manually supplied action table. Alpaca's current corporate-actions endpoint supplies split identifiers, symbols, rates, process dates, and ex-dates. It does not guarantee or expose the historical instant at which each record first became available. The endpoint documentation explicitly warns that provider receipt and processing can be delayed.

The implementation therefore needs a deterministic availability rule that cannot make a split known during an earlier part of its provider process date. It also needs an explicit factor convention compatible with ExecSim's existing adjustment operation, which divides prices by the factor and multiplies volumes by it.

## Decision

Acquire complete US `forward_split` and `reverse_split` records from Alpaca for every sourced symbol interval belonging to the frozen universe plus SPY. Preserve the paginated provider payload, normalized Parquet source, request identity, checksums, and an acquisition receipt. Resolve every action to exactly one stable instrument through the sourced symbol interval active on the ex-date; fail with `BLOCKED` instead of inferring an alias.

Store the split factor as:

```text
factor = old_rate / new_rate
```

This convention maps raw post-split observations to the preceding share basis when price is divided by `factor` and volume is multiplied by `factor`. Dollar notional remains unchanged.

Use 00:00 UTC on the calendar day after Alpaca's `process_date` as `available_at`. This is a conservative date-granularity surrogate: no observation on the process date can consume that action. Persist the surrogate policy and Alpaca's missing-creation-time limitation in the receipt. The provider snapshot remains a declared data-vintage limitation; the code does not present the surrogate as the actual announcement or API creation timestamp.

## Alternatives considered

- Treat `process_date` midnight as immediately available. Rejected because it can expose the action before Alpaca processed it during that day.
- Use the ex-date as availability. Rejected because effective date and knowledge date are distinct protocol identities.
- Infer actions from price jumps. Rejected because target observations cannot manufacture external metadata and the inference would be result dependent.
- Continue requiring a manual action file. Rejected because the historical pipeline would retain manual glue and could not reproduce its source boundary.
- Stop the entire program because creation timestamps are unavailable. Rejected because the frozen protocol already records provider vintage as a limitation, and the conservative next-day rule is deterministic, auditable, and does not grant earlier knowledge.

## Consequences

- Corporate-action acquisition is resumable and checksum-bound like market data acquisition.
- Same-day provider processing is never used by that day's market observations.
- An unmapped or ambiguous action blocks preprocessing instead of silently attaching to the wrong security.
- Historical revisions before the acquisition vintage cannot be reconstructed. Receipts state this limitation.
- The decision does not change the estimand, folds, universe, representation, model selection, or confirmatory comparisons. It implements the already frozen causal corporate-action boundary.

## Evidence

- [Alpaca corporate-actions API](https://docs.alpaca.markets/us/reference/corporateactions-1)
- Focused tests verify forward and reverse factor direction, next-day availability, pagination, idempotent reuse, dollar-notional preservation, and fail-closed symbol resolution.
