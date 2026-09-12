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
| [0012](0012-isolate-historical-sequence-members.md) | Accepted | Isolate historical sequence members in deterministic spawn-safe processes |
| [0013](0013-use-latest-causal-input-cutoff.md) | Accepted | Record the latest date used by any causal seasonal input |
| [0014](0014-cache-manifest-keyed-sequence-indexes.md) | Accepted | Cache sequence indexes by exact manifest identity |
| [0015](0015-complete-rdm-sweep-after-candidate-rejection.md) | Accepted | Complete the frozen RDM sweep after a candidate-level gate rejection |
| [0016](0016-separate-lightgbm-opencl-execution-identity.md) | Accepted | Separate LightGBM OpenCL runtime selection from the scientific grid and bind it to artifacts |
| [0017](0017-bound-historical-lightgbm-frame-construction.md) | Accepted | Bound historical LightGBM frame construction without changing rows or targets |
| [0018](0018-stream-historical-embedding-export.md) | Accepted | Stream historical embedding export with bounded memory and atomic failure cleanup |
| [0019](0019-freeze-observable-capacity-and-block-null.md) | Accepted | Freeze the secondary observable-capacity ladder and block-aware confirmatory null test before TEST |
| [0020](0020-bound-lightgbm-coordinate-memory.md) | Accepted | Bound LightGBM hybrid features and coordinate lifetime on the qualified host |
| [0025](0025-fail-closed-evaluation-input-identities.md) | Accepted | Fail closed on required ADV20 evidence and bind relocated evaluation identities before execution |
| [0026](0026-cross-link-resealed-evaluation-artifacts.md) | Accepted | Cross-link sequence, JEPA, embedding, and frozen LightGBM identities during reseal |
| [0027](0027-corporate-action-consistent-tca-adv20.md) | Accepted | Restate causal ADV20 into the target execution-share basis with point-in-time actions |
| [0028](0028-chained-resealed-evaluation-provenance.md) | Accepted | Preserve root TEST authorization and immediate predecessor across resealed executions |

## Recording a decision

ADR [0024](0024-promote-result-timestamp-precision.md) is accepted: promote only
timestamp precision during result merging, preserving instants and strict timezones.

ADR [0023](0023-record-unavailable-historical-forecasts.md) is accepted: record
method-specific historical forecast unavailability without changing estimators.

ADR [0022](0022-index-history-and-reseal-evaluation.md) is accepted: index historical
scope once and reseal optimized downstream evaluation without changing fitted models.

ADR [0021](0021-run-independent-lightgbm-folds-on-runpod.md) is accepted: run independent
LightGBM folds on three RunPod RTX 3090 pods with verified transfer and resumable artifacts.

ADR [0025](0025-fail-closed-evaluation-input-identities.md) is accepted: require
derived ADV20 evidence, bind relocated runtime universe bytes to every sequence,
and qualify JEPA source uniformity during reseal.

ADR [0026](0026-cross-link-resealed-evaluation-artifacts.md) is accepted: reject
 internally consistent but cross-coordinate-swapped immutable artifacts before
 publishing a resealed execution.

ADR [0027](0027-corporate-action-consistent-tca-adv20.md) is accepted: restate
causal ADV20 into the target execution-share basis using only point-in-time
known and effective corporate actions.

ADR [0028](0028-chained-resealed-evaluation-provenance.md) is accepted: bind a
new reseal to both the immutable root TEST authorization and the immediate prior
resealed execution with a typed supersession receipt.

ADR [0029](0029-reuse-immutable-evaluation-inputs.md) is accepted: reuse frozen
encoded probe batches, session-batched forecast features, and verified TCA date
slices without changing scientific ordering or mutable simulation state.

ADR [0030](0030-harden-operational-artifact-boundaries.md) is accepted: use exact
cache identities, restricted resume loading, uniquely staged receipts, parsed
child commands, and locked fresh CI dependencies without changing frozen inputs.

ADR [0031](0031-preserve-arithmetic-in-evaluation-fast-paths.md) is accepted:
index immutable preflight rows and use guarded forecast fast paths while retaining
the exact reduction order and general-case validation behavior.

ADR [0032](0032-enforce-evaluation-producer-contracts.md) is accepted: propagate
canonical evaluation identities and exercise real producer-to-consumer boundaries,
including report rendering and final completion verification.

Create an ADR when a change materially affects architecture, dependencies, mathematical formulation, information boundaries, artifact compatibility, or a performance design with non-obvious tradeoffs. Copy [the ADR template](0000-template.md), assign the next four-digit number, and add it to the index.

Do not rewrite an accepted decision to hide its history. Add a new ADR that supersedes it, then update the old record's status and link to the replacement. Update `docs/standards/implementation.md` in the same change when the decision changes the active direction.
