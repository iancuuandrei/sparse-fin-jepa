# ADR 0015: Complete the RDM sweep after candidate rejection

- Status: Accepted
- Date: 2026-09-06
- Scope: validation-only common-RDM selection orchestration

## Context

The frozen protocol requires all six Fold-1 candidates: three predeclared RDM coefficients crossed with dense and sparse geometry. A candidate is eligible only when both geometry checkpoints pass their collapse gates. The selection rule then minimizes the mean fixed-observable validation error across the eligible pair.

The trainer correctly rejected a sparse candidate after no validation checkpoint passed its frozen gate. The orchestration treated that expected candidate-level rejection as a fatal exception and stopped before testing the other predeclared coefficients. That behavior contradicted the frozen complete-six-run selection rule and could incorrectly turn one ineligible pair into a claim that no common coefficient is eligible.

## Decision

When no checkpoint in one candidate run passes its collapse gate, write an atomic `historical-training-rejection-v1` receipt with the exact fold, geometry, seed, RDM coefficient, configuration, sequence manifest, training configuration, code commit, completed steps, latest diagnostics, and gate reasons. Do not write or reuse a final checkpoint for that candidate.

The common-RDM orchestrator validates the rejection receipt, records the candidate as `FAIL`, and continues every remaining predeclared candidate. It runs the observable probe only for gate-passing checkpoints. Selection still requires the complete six-coordinate ledger and considers a coefficient only when both of its geometry rows are `PASS`. If no coefficient pair passes, fail without a parameter freeze.

## Alternatives considered

- Stop the sweep on the first rejected candidate. Rejected because it does not execute the frozen six-run matrix and cannot establish whether another predeclared coefficient is eligible.
- Loosen the sparse gate or change patience. Rejected because that would alter the protocol after observing development behavior.
- Assign a fabricated probe error to the rejected checkpoint. Rejected because no admissible checkpoint exists to probe.
- Omit the rejected coordinate from the receipt. Rejected because the selection contract requires a complete, auditable matrix.

## Consequences

- Candidate rejection remains fail-closed: no invalid weights can enter selection or final training.
- The predeclared development search completes even when one or more coefficient pairs are ineligible.
- Rejected rows carry no checkpoint hash and their placeholder numeric error is never used by the eligibility filter; the receipt carries an explicit failure reason.
- The decision changes no gate, patience, coefficient set, validation target, selection criterion, geometry, or estimand.

## Verification

- Selection tests cover a complete matrix containing one rejected sparse row and require the next lowest-error eligible pair to win.
- Gate-passing rows require finite non-negative probe errors and non-empty checkpoint hashes.
- A rejected candidate must carry a compatible receipt with at least one collapse-gate reason before orchestration continues.
