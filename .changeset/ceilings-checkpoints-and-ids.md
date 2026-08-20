---
"textopt": minor
"@textopt/langchain": patch
---

Ceilings hold where a run actually spends, checkpoints describe whole rounds,
and an instance id names one row.

Harvesting takes a `maxCostUsd` of its own. `harvestRollouts` and
`harvestFewShotExamples` check it between batches, and MIPRO and bootstrapped
few-shot search pass what is left of the run's ceiling into each pass. MIPRO
also stops building demo sets once the ceiling is reached. A demo menu is many
evaluations on a separate evaluator, so a ceiling it never consulted bounded
only the trial loop that followed it, and a run could spend its whole allowance
choosing demos and never score a candidate.

OPRO and MIPRO checkpoint after the sweep their cadence schedules, not before
it. A snapshot names a round, and a resumed run schedules its next sweep an
interval past the round the snapshot names, so a checkpoint taken first
described half a round and the resumed run skipped that sweep entirely. A MIPRO
trial whose rollouts all failed transiently now checkpoints and runs its cadence
like any other: the rollouts were bought and the counter moved either way.

Bootstrapped few-shot search reads every sweep it dispatched before it leaves a
wave. Stopping on the first failure abandoned the sweeps behind it, which went
on calling the adapter, and spending, after the caller had been handed the
error.

An adapter reading of `NaN`, `Infinity` or a negative token count is refused
where it enters. Folded into the totals it silently disabled `maxCostUsd`:
every later comparison against a `NaN` cost is false, so the ceiling stopped
holding without saying so. `createJudge` refuses a `scale` that is zero or
negative for the same reason — every grade is divided by it.

Instance ids fall back to the row's position for a datum a content hash cannot
read, not only for one that will not serialize. A `Map`, a `Set` and a class
instance holding its state privately all serialize to `{}`, so distinct rows
shared an id and were served each other's cached scores. The six optimizers now
share one `defaultInstanceId` rather than six copies of it.

A file cache terminates a record its previous process left half-written. The
truncated record was already lost; appending onto the line it left open lost the
next one too.

The LangChain adapter counts a provider that reports tokens in both shapes once.
Integrations that fill `llmOutput.tokenUsage` and `usage_metadata` for the same
call were billed twice, and the message-level shape now wins with the legacy
total as a fallback. Usage a scorer reports is no longer dropped when the run
itself counted none — the guard consulted only the callback total, so a judge's
spend went unreported.
