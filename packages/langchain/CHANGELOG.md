# @textopt/langchain

## 0.3.0

### Patch Changes

- a3354ac: The adapter marks a run or scoring failure it caught with `failed`, so the zero
  it stands in for is never written to the evaluation cache. `isTransient` still
  decides only what is worth retrying.
- 0aa5911: Every optimizer emits a `rollout` event carrying `completed`/`total` alongside
  the phase, split and candidate it belongs to, so a run reports progress between
  batches instead of going quiet for the length of a validation sweep. An adapter
  opts in by calling `args.onRollout` from its `evaluate` — the AI SDK and
  LangChain adapters already do, passing it as `onSettled` to
  `mapWithConcurrency`. A consumer switching exhaustively over an optimizer's
  event union must handle the new member.
- Updated dependencies [0aa5911]
- Updated dependencies [a3354ac]
- Updated dependencies [b62adf6]
- Updated dependencies [0aa5911]
- Updated dependencies [12789b5]
- Updated dependencies [0aa5911]
- Updated dependencies [0aa5911]
  - textopt@0.3.0

## 0.2.0

### Patch Changes

- Updated dependencies [c2c4d3e]
- Updated dependencies [c2c4d3e]
- Updated dependencies [c2c4d3e]
- Updated dependencies [c2c4d3e]
- Updated dependencies [c2c4d3e]
- Updated dependencies [3f5824b]
  - textopt@0.2.0

## 0.1.0

### Minor Changes

- 155cb19: First public release: GEPA, SIMBA, OPRO, MIPRO, bootstrapped few-shot search,
  and random search behind a shared optimizer interface, with a LangChain
  adapter. The shared substrate handles budgets in rollouts, dollars and wall
  clock, transient-failure retry, durable caching, checkpoints and resume for
  every optimizer, a model-graded judge, a multi-module pipeline adapter, and
  `compare()` for deciding between two runs with a p-value rather than a hunch.
  Every run reports through `reporters`: any number of observers, each with an
  `onEvent` called synchronously and a `flush` awaited as the run ends, including
  when it ends by throwing. Each search emits its own event union, but every one
  of them announces an accepted candidate with the text that scored and its row
  over the validation set, and reports the held-out sweep per instance rather
  than as a lone mean. A reporter that reads only those narrows with
  `isCandidateAccepted` and `isRunFinished` and drops into any optimizer.

  The Vercel AI SDK, Braintrust and LangSmith packages are still in beta and ship
  from the repository only.

### Patch Changes

- fec8f51: Ceilings hold where a run actually spends, checkpoints describe whole rounds,
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

- 38dfb12: A ceiling bounds what it was told it bounds, and a reading it cannot be checked
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

- Updated dependencies [ca8a541]
- Updated dependencies [b25abd2]
- Updated dependencies [fec8f51]
- Updated dependencies [b25abd2]
- Updated dependencies [155cb19]
- Updated dependencies [3336175]
- Updated dependencies [38dfb12]
- Updated dependencies [7809e1d]
- Updated dependencies [2606e36]
- Updated dependencies [b2be4d6]
  - textopt@0.1.0
