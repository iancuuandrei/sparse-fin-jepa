# ADR 0033: Require canonical transfer member identities

Status: accepted

## Context

Transfer manifests reject repeated member names, but two different strings can
resolve to the same file. `PurePosixPath` normalizes redundant separators and
dot components. Checking duplicates before rejecting those aliases does not
establish a unique file identity. The former symbolic-link check also covered
only the final component, not a directory component within the member path.

## Decision

Require each member to be a nonempty canonical relative POSIX path. Reject dot
aliases, repeated separators, trailing separators, NUL characters, parent
traversal, absolute paths, and Windows separators or drive syntax. Reject
symbolic links in every component below the selected root, then retain the
resolved containment check. The operator-selected root remains relocatable.

Keep frozen numeric configuration checks exact. Group expected fields by
configuration section to make the locked contract inspectable; do not introduce
approximate comparisons or change configuration serialization.

## Alternatives considered

Silently normalizing manifest names conceals ambiguous inventory entries.
Restricting operator-selected roots to the repository breaks portable imports.
Suppressing analyzer findings would not strengthen the member boundary.

## Consequences

Canonical producer-generated packages remain compatible. Noncanonical member
names now fail before payload access. Existing immutable artifacts are not
rewritten. Scientific inputs, estimands, and model behavior do not change.
These checks assume a trusted local staging directory that is not concurrently
mutated; they do not claim protection against an adversarial filesystem race.
