# ADR 0037: Chain inherited stage producers across recoveries

Status: accepted

## Context

Execution e8d047b inherited completed forecast and representation stages from
6402e40, then failed in TCA. Its replacement must name e8d047b as the immediate
predecessor while continuing to consume the original 6402e40 stage bytes.
The first inheritance schema assumes that the immediate predecessor produced
both stages. Reusing that assumption either fails on absent files or incorrectly
skips an execution in the provenance chain.

## Decision

Retain direct-producer `paper-evaluation-stage-inheritance-v1` receipts unchanged.
Use `paper-evaluation-stage-inheritance-v2` when the predecessor already inherited
the stages. Keep immediate supersession fields distinct from explicit per-stage
producer namespace, execution receipt checksum, and evaluator commit/tree.
Bind the predecessor's inheritance receipt by path and checksum. Verify its
native execution and inheritance chain and require the new producer inventory
to equal that verified ancestor inventory.

Only forecast and representation may be inherited, and the invalidation frontier
remains TCA. The resolver returns the verified original producer path and source,
not the immediate predecessor's empty result directory. Report and final freeze
record the same actual producer. Execution v4 already binds the typed inheritance
receipt, so its field meanings remain unchanged. The supervisor accepts both
explicitly supported inheritance schemas only after native verification.

Cache validity includes ancestor receipts and inherited members. A changed
ancestor must invalidate a previously successful verification. A missing or
different inventory fails closed. No migration, copying, relabeling, old TCA
reuse, or arbitrary stage DAG is introduced.

## Alternatives considered

Naming 6402e40 as the immediate predecessor would erase e8d047b. Copying stages
into e8d047b would mutate evidence and misstate authorship. Recomputing the
unaffected stages would discard the explicitly authorized recovery benefit.
Changing v1 field meanings in place would make old receipts ambiguous.

## Consequences

The chain retains both execution succession and original stage authorship.
New TCA, report, and final-freeze outputs remain bound to the replacement source.
Scientific inputs, models, populations, estimands, and authorization scope are
unchanged. ADR 0035 remains applicable; this extends it to later generations.
Tests must exercise native reseal, recursive mutation rejection, actual report
construction, and final freeze with an intervening inherited execution.
