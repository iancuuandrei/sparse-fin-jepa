# ADR 0023: Record unavailable historical forecasts by request

- Status: Accepted
- Date: 2026-09-11
- Complements: [ADR 0022](0022-index-history-and-reseal-evaluation.md)

## Context and evidence

The v2 token-valid sample can contain missing provider minutes. The unchanged
historical EWMA estimator requires complete preceding minute windows. During the
203c3ee execution, the BKNG 2024-04-12 10:30 full-session request had no eligible
window among 571 preceding sessions. Timestamp coverage established this condition;
no comparative model errors informed this decision. The worker propagated this
legitimate unavailability as a fatal error and stopped independent methods.

## Decision

Raise a specific ValueError subclass only for absent eligible historical data.
Paper EWMA workers record each unavailable exact request with sample, instrument,
date, as-of, requested end, and reason. Continue other requests and instruments.
Do not impute bars, change the estimator, substitute another provider, remove an
instrument globally, or catch unrelated validation and software exceptions.

Ledger schema v4 includes a checksummed availability file. Full-session and TCA
windows have independent availability. TCA records a method-specific unavailable
case if a required request is unavailable; no partial replay is reported as a
completed result. Side assignment and other providers are unchanged.

Pairwise statistics remove explicitly unavailable rows before intersection and
count them in dropped-case receipts. Non-finite available metrics still fail.
Comparisons with no common cases produce an unavailable-comparison receipt, not
invented intervals. Confirmatory comparisons retain pair-specific populations;
the existing descriptive all-method table remains common-case with coverage counts.

## Alternatives considered

Failing the pipeline prevents independent work. Filling absent minutes with zero
invents observations. A token-level replacement or different lookback changes the
estimator. Silently dropping a symbol changes other comparisons. All are rejected.

## Scientific consequences

The estimator, targets, selections, scientific configuration, and paired estimand
are unchanged. An EWMA comparison describes only common available cases, not all
token-valid sessions. Dense/sparse confirmatory comparisons do not inherit EWMA
exclusions. Missingness may be systematic; coverage must accompany interpretation.

## Compatibility and verification

Old v3 ledgers cannot masquerade as availability-aware v4 ledgers. A new downstream
source seal is required; upstream models remain immutable. Tests cover partial
and empty availability, distinct horizons, resume/checksum rejection, unrelated
exceptions, TCA method isolation, and finite paired statistics.
