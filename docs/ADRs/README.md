# Architecture decision records

Architecture decision records (ADRs) explain durable technical choices, their context, and their consequences. They complement the current rules in `docs/standards/implementation.md`: the standard says what applies now, while ADRs explain why the repository adopted that direction.

## Decision index

| ADR | Status | Decision |
|---|---|---|
| [0001](0001-python-research-stack-and-manifest-navigation.md) | Accepted | Use the Python research stack and manifest-backed navigation instead of Nx |
| [0002](0002-point-in-time-policy-boundary.md) | Accepted | Enforce point-in-time policy inputs and exogenous historical replay |
| [0003](0003-explicit-convex-qp-and-linear-impact.md) | Accepted | Use an explicit OSQP convex program and linear-in-participation temporary impact |
| [0004](0004-forecast-only-ml-boundary.md) | Accepted | Restrict V1 ML to point-in-time input forecasts |
| [0005](0005-causal-forecast-cache-and-horizon-workspaces.md) | Accepted | Cache causal forecast matrices and reuse horizon-indexed OSQP workspaces |
| [0006](0006-sparse-predictive-representation-paper-framework.md) | Superseded | Add a controlled sparse predictive-representation paper framework |
| [0007](0007-harden-historical-paper-pipeline-contracts.md) | Superseded | Harden the historical paper pipeline contracts and orchestration |
| [0008](0008-redirect-paper-to-representation-accessibility.md) | Accepted | Redirect the paper to representation accessibility |
| [0009](0009-separate-data-quality-by-resolution.md) | Accepted | Separate formation, representation, and execution data quality by resolution |
| [0010](0010-separate-runtime-approval-from-scientific-config.md) | Accepted | Separate runtime stage approval from scientifically frozen configuration |
| [0011](0011-source-split-actions-with-conservative-availability.md) | Accepted | Source split actions with conservative provider-process-date availability |

## Recording a decision

Create an ADR when a change materially affects architecture, dependencies, mathematical formulation, information boundaries, artifact compatibility, or a performance design with non-obvious tradeoffs. Copy [the ADR template](0000-template.md), assign the next four-digit number, and add it to the index.

Do not rewrite an accepted decision to hide its history. Add a new ADR that supersedes it, then update the old record's status and link to the replacement. Update `docs/standards/implementation.md` in the same change when the decision changes the active direction.
