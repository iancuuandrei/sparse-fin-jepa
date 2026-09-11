# ADR 0026: Cross-link resealed evaluation artifacts

- Status: Accepted
- Date: 2026-09-12
- Deciders: ExecSim maintainers

## Context

The evaluator reseal already checks the checksums of sequence manifests,
LightGBM models, JEPA checkpoints, and embedding exports.  Independent
checksum validation is not sufficient: a complete but incompatible artifact
set can otherwise be sealed when the artifacts point to different sequence,
universe, representation-source, or embedding inputs.

## Decision

During reseal, validate the existing cross-links for every configured fold and
representation coordinate:

- every LightGBM manifest names the current fold sequence manifest;
- every JEPA checkpoint and embedding export names that same sequence;
- JEPA checkpoint universe identity matches the sequence universe identity;
- checkpoint and compatibility metadata retain the same sequence identity;
- embedding normalization and paper configuration identities match the
  checkpoint and evaluator configuration;
- LightGBM hybrid coordinates name the exact TRAIN and VALIDATION embedding
  bytes that are being sealed;
- each JEPA checkpoint source commit matches the representation source commit
  recorded by the parameter freeze.

These checks run inside `frozen_inventory()` before `execution.json` is
published.  Existing final-freeze checks remain as defense in depth.

## Consequences

An internally consistent but cross-coordinate-swapped checkpoint, embedding,
sequence, or LightGBM artifact is rejected before any derived evaluation work
starts.  The immutable artifacts, scientific configuration, estimands, and
model mathematics are unchanged.  The evaluator source commit remains
distinct from the immutable representation training commit.
