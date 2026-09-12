# ADR 0027: Restate causal ADV20 into the execution-share basis

- Status: Accepted
- Date: 2026-09-12
- Complements: [ADR 0011](0011-source-split-actions-with-conservative-availability.md), [ADR 0025](0025-fail-closed-evaluation-input-identities.md)

## Context

The locked TCA quantity is an order size in shares, and replay bars and
participation capacities are also expressed in shares. A split inside the
prior 20-session history makes a raw rolling share-volume mean mix incompatible
share units. The existing point-in-time split convention is already frozen for
sequence features: a cumulative factor `F` divides prices and multiplies
volume, and an action is usable only when effective and known at the case
information time.

## Decision

For a target execution session `t` and a prior eligible session `h`, compute

`V_h,t = V_h * F(h, I_t) / F(t, I_t)`

where `I_t` is the frozen 10:30 America/New_York quantity decision and `F` is
resolved with the existing effective-time and `available_at` rules. The
20-session lag remains strict: target-session volume is never included. Replay
bars remain raw target-session bars, so ADV20, parent quantity, and replay
capacities share one target-session execution basis. Corporate-action manifest
bytes and checksum are part of the TCA-history identity; stale histories are
rejected.

## Consequences

Known effective actions can restate prior volumes; announced-but-not-effective
or effective-but-unknown actions cannot. Multiple actions compose through the
existing cumulative product. No scientific population rule changes: exact
TCA-window eligibility is still resolved before ADV validation. A new history
identity schema is used for action-bound histories, while old no-action fixture
histories remain readable for compatibility tests.
