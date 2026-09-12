# ADR 0031: Preserve arithmetic in evaluation fast paths

Status: accepted

## Context

Immutable forecast and ledger rows repeatedly passed through pandas grouping,
filtering, merges, and timestamp dictionaries. These costs are distinct from
LightGBM inference, MPC optimization, and the frozen statistical estimators.
Removing them must not reorder floating-point reductions or weaken rejection
of invalid populations.

## Decision

Index learned preflight frames by date and case once per instrument. Keep the
physical row order within each selected shape slice and run all existing
identity, cutoff, duplicate, origin, and conditional-share checks. No worker
starts until preflight succeeds.

For contiguous non-null, noncategorical shape groups, discover group boundaries
once and assign softmax output with NumPy positions instead of pandas writes.
Keep each group's original max, exponential, and sum operations and ordering.
Retain the old pandas path for unsupported or interleaved keys.

Bypass the metric outer merge and sort only after proving identical typed,
unique keys, contiguous cases, and ascending target buckets. Preserve all metric
reductions. The fallback checks share sums before sorting, exactly as before.

At the two learned-provider minute-grid construction sites, retain immutable
grid metadata. Truncation may slice volumes only when the cached timestamp tuple
is the trusted grid object and the complete request equals a contiguous grid
slice. Otherwise use the existing dictionary behavior, including last-key wins
for duplicate generic timestamps. Read volumes afresh; do not retain a stale
timestamp-to-volume dictionary. Normalization and output construction are unchanged.

## Alternatives considered

Segmented approximate reductions or a different summation order could change
probe or metric trajectories and rejection thresholds. Generic sorted-timestamp
assumptions are unsafe because `VolumeForecast` does not require sorted unique
timestamps. Reusing DataLoaders across coordinates could advance a shared RNG
generator differently. These alternatives are not adopted.

## Consequences

The fast paths preserve scientific inputs, fitted models, partitions, estimands,
and exact arithmetic where the reference is deterministic. Synthetic parity
tests include irregular keys, invalid populations, every minute of a 300-minute
window, and full replay equality. Benchmarks report their bounded scope rather
than extrapolating a historical completion time.

Truncation still constructs a forecast for every requested remaining minute;
the optimization removes dictionary overhead, not the output-size lower bound.
Hash verification is unchanged and old execution results are never reused.
