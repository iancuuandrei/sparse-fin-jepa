# Documentation guide

Start with the completed experiment, then follow the scientific or engineering
references needed for your task. Operational procedures are not instructions to
restart the completed study.

## Understand the research

- [Experiment results and provenance](EXPERIMENT_RESULTS.md): completion, frozen identities, confirmatory results, and limitations.
- [Paper design](PAPER_DESIGN.md): research question, models, folds, targets, and predeclared inference.
- [Implementation report](PAPER_IMPLEMENTATION_REPORT.md): current status and dated development evidence.
- [Research references](RESEARCH_REFERENCES.md): scientific and provider sources.
- [Point-in-time contract](DATA_LEAKAGE_CONTRACT.md): information clocks and prohibited future information.
- [Mathematical model](MATHEMATICAL_MODEL.md): execution objectives, units, fills, and costs.

## Understand the implementation

- [Repository navigation](NAVIGATION.md): component ownership and context commands.
- [Specifications](SPECIFICATIONS.md): executable component contracts.
- [Evaluation implementation](EVALUATION_IMPLEMENTATION.md): artifact validation, stage boundaries, recovery, and final freeze.
- [Architecture decisions](ADRs/README.md): durable decisions and their rationale.
- [Implementation standard](standards/implementation.md): code and documentation conventions.

## Operate a separately authorized run

- [Runtime authorization](PAPER_RUNTIME_AUTHORIZATION.md): distinct acquisition, training, and evaluation permissions.
- [RunPod deployment](RUNPOD_EXECUTION.md): portable fold training and qualification.
- [Repository README](../README.md): installation, offline simulation, and safe planning commands.

The frozen result bundle is external to Git. Documentation updates do not change
its producing commits, grant execution approval, or replace checksum verification.
