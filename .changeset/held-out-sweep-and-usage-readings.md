---
"textopt": minor
"@textopt/langchain": patch
---

A ceiling bounds what it was told it bounds, and a reading it cannot be checked
against is refused wherever it enters.

The held-out sweep is reported apart from the search. `maxCostUsd` and
`maxWallClockMs` bound the search loop, and a `testSet` is measured once after
that loop has already stopped — charging it would let the size of a held-out set
decide which candidate wins. `metricCalls` already excluded those rollouts and
reported them as `testMetricCalls`; `usage` did not, so a run that honoured a $4
ceiling and then swept a hundred held-out rows reported spending $104 and looked
like it had overrun. Their dollars are now `testUsage`, alongside
`testMetricCalls`, in all six optimizers. SIMBA and bootstrapped few-shot search
count `testMetricCalls` the way the other four already did — the rollouts the
sweep bought, not the rows it was handed, which differ once one is retried or
served from the cache.

`Evaluator` gains `unchargedUsage()` for the same split, and `usage()` now
returns only what the search spent. Nothing bounds the held-out sweep, so what a
run costs end to end is `usage` plus `testUsage`: budget for a `testSet` the way
you would budget for one full validation sweep.

A usage reading that is not a non-negative finite number is refused whether it is
a number or not. The guard only fired on `NaN`, `Infinity` and negatives, so a
JavaScript adapter reporting `costUsd` as a string concatenated onto the totals
and left the ceiling checked against text. Resumed usage and usage absorbed from
a harvest are checked too: a hand-edited checkpoint could poison the totals with
nothing to say it had.

The LangChain adapter reads the legacy total again when the message-level shape
counts nothing. Preferring `usage_metadata` on its presence rather than on what
it carries meant an integration attaching a zeroed one — as some do to
intermediate generations — reported zero tokens for a call whose real total was
sitting in `llmOutput.tokenUsage` all along.
