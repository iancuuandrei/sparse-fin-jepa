# ADR 0030: Harden operational artifact boundaries

Status: accepted

## Context

The encoded probe cache and published representation coordinate use different
schemas. Comparing the coordinate schema during cache retirement rejects a valid
cache after publishing the coordinate. Ignoring that field would weaken identity
verification instead of fixing the mismatch.

The operational audit also found unrestricted trusted-resume deserialization,
predictable temporary receipt names, and a detached launcher that replaced the
first command-like string in raw arguments, including a directory named `start`.

## Decision

Use one complete encoded-cache identity constructor for materialization and
retirement. Bind the source, coordinate, checkpoint, sequence, batching,
partition, device, and PyTorch version. Verify all partitions and checksums
before removing any cache file. Preserve coordinate publication independently.

Read resume bytes once, verify their checksum, and deserialize those same bytes
with PyTorch's restricted weights-only loader. Scope the additional NumPy types
to those required by the existing MT19937 state. Keep explicit trusted-local
authorization; do not rewrite historical checkpoint bytes.

Publish JSON receipts through uniquely created temporary files in the destination
directory. Flush and synchronize the file before atomic replacement. On failure,
remove only this invocation's temporary file; retain the previous receipt and
unrelated temporary evidence. This does not promise directory-entry durability
across every filesystem or host failure.

Construct detached child commands from parsed fields with a fixed interpreter,
script, and operation. Pass each option and value as one argument without a
shell. Require positive integer process identities before reading Linux process
metadata. Preserve explicit user-selected data and output roots.

Lock fresh CI installations with `uv.lock` and immutable action revisions. This
does not authorize replacing the qualified historical environment or its custom
LightGBM OpenCL build.

## Alternatives considered

Ignoring cache schemas conceals incompatible identities. Unrestricted pickle
loading is unnecessary for the persisted optimizer/RNG contract. Restricting all
CLI paths to the repository would break runtime relocation; artifact members
must instead remain within their explicitly chosen authoritative roots.

## Consequences

Scientific configuration, model weights, sample populations, probe trajectories,
and estimands remain unchanged. Frozen numeric configuration is compared exactly:
even a one-ULP parameter change fails, rather than passing an approximate check.

Regression tests cover production-schema cache retirement, identity corruption,
no partial deletion, restricted resume compatibility, failed receipt publication,
and operation-like directory names. Unsupported resume objects fail closed.
