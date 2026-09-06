# ADR 0013: Use the latest causal input cutoff

- Status: Accepted
- Date: 2026-09-06
- Scope: paper sequence provenance and seasonal features

## Context

A sequence can follow a provider gap longer than 20 trading days for its stock while SPY remains valid. The stock seasonal history then ends on the stock's last valid session, but the independently causal SPY seasonal window ends on the exchange session immediately before the current case.

The corpus builder previously declared the stock's last valid date as the record cutoff and passed that date to both seasonal baselines. When every member of the 20-session SPY window was newer than that stock date, the baseline correctly rejected the contradiction: the supplied SPY history was not available by the declared cutoff. Treating the rejection as an exclusion would discard a token-valid current stock session because of a provenance-label defect rather than unavailable causal inputs.

## Decision

Declare each sequence record's feature-history cutoff as the latest session date used by any causal seasonal input. In the current architecture that is the final SPY session strictly before the current session. Keep the stock seasonal window on its own last 20 valid observations and keep `previous_close` on the stock's most recent valid observation.

Both stock and SPY seasonal rows must be on or before the declared cutoff. Neither input may include the current session or a future session.

## Alternatives considered

- Keep the stock date and discard newer causal SPY rows. Rejected because it unnecessarily removes information available before the decision and changes SPY windows after stock data gaps.
- Exclude the current stock session. Rejected because the session and its causal inputs are valid; only the provenance label was inconsistent.
- Store the current session date as cutoff. Rejected because it is less precise and could obscure accidental current-session seasonal input.
- Add separate stock and SPY cutoff fields during the frozen run. Rejected because the existing maximum-feature-date field can represent the compatibility boundary without changing the sequence schema.

## Consequences

- Long stock gaps no longer make a valid SPY baseline appear to violate its declared cutoff.
- The record cutoff remains a conservative maximum: every seasonal observation is at or before it.
- Stock `previous_close` and stock seasonal history can be older than the record cutoff; their actual observations remain unchanged.
- The decision changes no fold, feature formula, inclusion threshold, target, model, estimand, or locked comparison.

## Verification

- A regression test constructs a stock gap longer than the SPY window, verifies successful construction, and requires the record cutoff to equal the latest prior SPY session.
- Existing leakage and fold tests continue to reject current-session and future-session history.
