# TCA numerical recovery qualification

This record concerns implementation and artifact provenance, not scientific
effectiveness. The preserved `6402e40` evaluation is a failed partial execution.
Its forecast and representation stages remain attributed to their original source.

## Starting identity and reproduced failure

- Source: `6402e40d42d8ff5f5e09d77611e23f3646d46ca7`.
- Tree: `84b4d490984978ad0a8661a50c1a9ad03eefb336`.
- Scientific configuration: `e0a926869b8882ff3a3f031bda046786ccaf9194ad9a6f1e4a5b7c9846609ed1`.
- Preserved execution receipt: `0ec600bea399906fca92dfc669b812e936c2524580c3b5ca63913b35cf5c9ee0`.

The original source reproduced the failure outside its execution namespace before
the optimizer was edited. The case was fold-1, 2024-04-01, instrument
`sec-cik-0000080424-PG`, dense seed 47, 5% ADV sensitivity, parent quantity 196,101.
At 11:30 America/New_York, the remaining quantity was 155,641 over 240 minutes;
integer capacity totaled 155,652.

OSQP reported `solved` after 445 iterations. Its largest upper-bound residual was
0.0077272608798466536 shares; the completion residual was approximately
-0.0000039351 shares. The configured absolute and relative tolerances were both
`1e-7`. The application-side absolute-plus-relative share budget is approximately
0.0155642 for this constraint scale. The old integer boundary instead rejected any
box residual above `1e-5`. This was a producer/consumer tolerance mismatch, not an
infeasible parent order.

An adversarial 300-interval regression also covers accumulated lower-bound
clipping: 200 residuals of -0.007 shares can create 1.4 shares of aggregate excess.
The sanitizer repairs that otherwise-failing strict-floor branch using the exact
nonnegative target-sum projection, after raw feasibility has already passed.

The new boundary validates raw completion with explicit `rtol=0`, validates raw
box residuals against the configured share-unit budget, and clips admissible box
residuals before integer reconciliation. The integer epsilon remains `1e-5`;
solver settings, costs, and scientific quantities are unchanged. Unique feasible
cases bypass OSQP. See [ADR 0034](ADRs/0034-separate-solver-acceptance-from-integer-rounding.md).

## Recovery contract matrix

| Boundary | Preserved invariant | Qualification |
| --- | --- | --- |
| OSQP → integer schedule | Original-unit acceptance, fixed rounding epsilon, exact target and bounds | Deterministic cold/reused/shrinking workspace stress and retained failing problem |
| MPC failure → process worker | Case identity and original exception survive process serialization | RuntimeError/ValueError and pickle regressions |
| Completed forecast → inherited inputs | Profile, bases, learned/EWMA ledgers and merged files retain producer source and byte identity | Typed inventory validation and mutation regressions |
| Completed representation → inherited inputs | Configured coordinate inventory, merged outputs and upstream cross-links | Typed inventory and real mixed-source report fixture |
| Inherited inputs → fresh TCA | Real providers/preflight/replay read old source; all new TCA artifacts use replacement source | Production-shaped learned/EWMA TCA stage regression |
| Mixed numerical stages → report | Source per stage and exact input hashes remain explicit | Real report builder, historical bundle, and resume/corruption tests |
| Report → final freeze | Completion/provenance, upstream inventory and mixed-source stage hashes agree | Real final-freeze integration |
| Prior execution → new seal | Root authorization and immediate predecessor remain distinct | Typed execution-v4 inheritance and recursive verification |
| Seal → detached supervisor | Only TCA, report, final freeze may run; failure stops later stages | Supervisor dispatch, preflight-failure and graceful-signal tests |

The inheritance contract is deliberately limited to forecast and representation;
it cannot import old TCA shards. See
[ADR 0035](ADRs/0035-inherit-completed-evaluation-stages-explicitly.md).

## Qualification status

Final local gates, exact-head CI, read-only pod qualification, and detached launch
must be recorded against their actual source identities before recovery is declared
complete. A passing fixture does not mean historical TCA or final results passed.

The complete failed namespace was inventoried before changes: 3,597 regular files,
22,128,283,655 logical bytes. The external preservation inventory records each
file's SHA-256, size, inode and modification timestamp. Recovery must not mutate
those bytes or reuse the 28 completed old TCA shards.
