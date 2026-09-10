# Data leakage contract

Point-in-time integrity is a hard correctness property, not a reporting preference.

## Information clock

| Time | Permitted information |
|---|---|
| Before session | Prior-session bars and derived features whose `available_at` is no later than the pre-session cutoff; calendar and externally supplied assumptions |
| Start of bucket `k` | Completed buckets strictly before `k`; forecasts generated at or before this timestamp; remaining inventory and prior fills |
| While bucket `k` forms | For POV only, contemporaneously materializing volume may determine participation quantity; the eventual close/high/low/VWAP/total volume is not available as a decision feature |
| After bucket `k` closes | Final bar fields for `k` may update the next decision, forecast, and diagnostics |
| After session | Full-session targets and benchmarks may be constructed for evaluation, never retroactively supplied to deployable policies |

## Enforceable boundaries

- `DecisionContext` contains observations through a declared cutoff and forecasts for the future; it never contains the full target-day DataFrame.
- Static forecasts have `training_data_cutoff < session_date`.
- Dynamic samples state `as_of`; every feature record satisfies `available_at <= as_of`.
- Global chronological split dates apply across all symbols. No symbol/date belongs to more than one partition of a fold.
- Preprocessors fit on training partitions only. Validation may select hyperparameters. Test partitions remain locked until selection completes.
- Historical-profile VWAP receives prior sessions separately and rejects target/future dates.
- `HistoricalProfileForecaster` indexes history once per symbol or pooled scope. Its
  internal read-only matrix may contain later sessions; binary search selects only
  dates strictly before the requested session before inspecting window completeness.
  Stable earliest-timestamp row order is preserved, including pooled ties. Requested
  buckets must all be finite; the last configured number of complete rows is selected
  before zero-total rows are removed. Mean, median, previous, and finite-window EWMA
  mathematics, warnings, and cutoffs are unchanged. There is no per-date prefix cache.
- Oracle policies are type- and registry-separated, visibly labeled, and excluded from deployable defaults.
- Paper model parameters, normalization, category vocabularies, regime thresholds, and within-bin profiles fit on TRAIN only. Held-out ADV20, previous close, seasonal histories, and known/effective corporate-action state update causally through each case's `feature_history_end` and `market_information_as_of`; they are not model parameters. The JEPA encoder receives 13 dynamic fields and no symbol identifier; five clock/scale fields condition its predictor only. Exported embeddings contain the current observed latent, predicted future latents, and explicit availability flags, never encoded actual future tokens.
- The v2 formation liquidity rank is computed only after the fixed 2021 formation interval and is never described as known at its start. Target-period gaps cannot replace or remove a frozen universe member. Daily formation validity, observed-only 15-minute representation validity, and exact consumed-window TCA validity are separate identities; converting a missing provider minute into a zero-volume observation would fabricate information and is prohibited. Target-period symbols are resolved point in time from sourced stable-identity intervals and verified against the session bars; the formation symbol cannot silently label a renamed security.
- Locked-test forecast, representation, and TCA stages require a checksummed parameter-selection freeze produced from validation artifacts. Missing empirical artifacts are `NOT RUN` or `BLOCKED`; synthetic output cannot substitute for a historical result.
- Paper acquisition requires Alpaca SIP. Missing entitlement is an error; the system cannot substitute IEX or another feed.

## Bar-level POV convention

The simulator treats POV quantity as flow that participates while trades occur inside a minute. It may use the volume realized during that minute only for the mechanical quantity `floor(rho*volume)`; it may not use that minute's eventual price path or any later bar to choose aggressiveness. The fill is summarized at the bar reference price after the bucket closes. This is an aggregation convention, not a claim that the closing bar was known beforehand.

## Failure behavior

Future-dated features, forecasts generated after their first forecast bucket, overlapping split memberships, missing `as_of`, ambiguous timezone, or target-day profile leakage cause validation errors. Missing buckets are excluded or masked with a recorded reason; they are never silently filled with zero.
