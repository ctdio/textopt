# textopt

## 0.3.0

### Minor Changes

- a3354ac: A score an adapter synthesized after catching an error is never written to the
  evaluation cache. Adapters say so with `failed` on `ScoreResult`, which takes no
  judgement about the provider; `transient` still decides what is retried and kept
  out of a candidate's mean, and still classifies nothing by default. A run that
  finished with failures nothing classified reports how many under the new
  `unclassifiedFailures` warning code.

  A repeat run over a warm `file-cache` that hit failures now re-runs those
  instances rather than reading their zeros back, so it can spend more rollouts
  and reach a different winner than the run before it.

- 12789b5: A `reflect` call that throws is retried under the run's existing `retry`
  policy, rather than ending the run. Nothing is classified on the way past: a
  proposal model is a pure request, so the attempt after a transport failure is
  free to succeed and a genuine bug fails `attempts` more times and surfaces
  unchanged. Every attempt counts against the optimizer's reflection ceiling —
  GEPA's `reflection.maxCalls`, OPRO's and SIMBA's `maxReflectionCalls` — which a
  round can now overrun by up to `attempts`. Set `retry: { attempts: 0 }` to keep
  the old behaviour. `withRetries(model, policy)` is exported for applying the
  same policy to any other `TextModel`.
- 0aa5911: Every optimizer emits a `rollout` event carrying `completed`/`total` alongside
  the phase, split and candidate it belongs to, so a run reports progress between
  batches instead of going quiet for the length of a validation sweep. An adapter
  opts in by calling `args.onRollout` from its `evaluate` — the AI SDK and
  LangChain adapters already do, passing it as `onSettled` to
  `mapWithConcurrency`. A consumer switching exhaustively over an optimizer's
  event union must handle the new member.

### Patch Changes

- 0aa5911: `candidateAccepted` carries `objectiveScores`, the per-objective mean over the
  instances the candidate was measured on. A single objective collapsing while the
  aggregate holds is how a degenerate metric channel announces itself, and it was
  previously visible only in the result.
- b62adf6: `createPromptAdapter({ run, score })` from `textopt/gepa` is the single-prompt
  case named: `run` receives the candidate's text as `instruction` and `score`
  grades the output. It reads which component to run off the candidate and throws
  when there is more than one, because a component no module runs is text the
  search rewrites every iteration for no effect — that system wants
  `createPipelineAdapter`. Neither helper is GEPA-only, and the docs now say so:
  a `GepaAdapter` is the base `Adapter` with reflection's evidence added, so the
  same adapter passes unchanged to SIMBA, OPRO, MIPRO and both searches.
- 0aa5911: `createReporter({ on })` takes a handler map whose keys are checked against the
  optimizer's event union, so a misspelled event name is a compile error rather
  than a reporter that runs to completion having seen nothing. A reporter built
  this way also declares what it handles, and a run warns at start about a
  handler named for an event no optimizer emits, or about a reporter whose
  handlers all miss the run it is attached to. A handler for an event some other
  optimizer emits is not warned about: one reporter written for several searches
  is what the shared events are for.
- 0aa5911: Two reporters ship with the library: `consoleReporter({ level })` prints one
  line per event — `quiet` for acceptances and the finish, `verbose` for
  everything — and `jsonlReporter({ path })` from `textopt/file-reporter` appends
  each event as a JSON line, leaving structured data structured instead of
  flattening it into log prose.

## 0.2.0

### Minor Changes

- c2c4d3e: `GepaOptimizer` refuses an acceptance policy that cannot accept anything at the
  configured `minibatchSize`. A sign-flip test over three instances bottoms out at
  p = 0.125, so `pairedPermutationAcceptance({ alpha: 0.05 })` at the default
  minibatch rejected every proposal and returned the seed after spending the whole
  budget. Acceptance policies report the batch they need as `minimumPairs`; raise
  `minibatchSize` to at least that, or loosen `alpha`.
- c2c4d3e: `createFileCache` requires a `namespace` naming the system its scores measure,
  and never serves an entry written under a different one. A durable log outlives
  the model behind an alias, the decoding settings, and the scorer version — every
  part of a measurement a cache key does not name. Pass the same string you would
  pass as `cacheNamespace`.
- c2c4d3e: `GepaAdapter.evaluate` returns a `ReflectiveBatch` — an `EvaluationBatch` with
  `feedback` required. An adapter that returned scores and no prose left the
  reflection prompt rewriting instructions from empty feedback blocks, which is
  blind search reported as a normal run. Adapters that already return `feedback`
  need no change; the rest now fail to compile.
- c2c4d3e: `JudgeCriterion` takes `weight` and `gate`. A gated criterion that grades below
  its bar scores the instance 0 whatever the other criteria said, so a hard
  requirement is no longer something a search can trade away against three
  cosmetic ones. When `expected` is passed, the default judge prompt also forbids
  the feedback from restating it — feedback is rewritten into a reusable
  instruction, and a fact copied out of the gold answer becomes an answer key
  memorised in the prompt.
- c2c4d3e: Every result and `finish` event carries `warnings`: what a run could see about
  its own measurement that its numbers cannot say. A run given no `validationSet`
  reports that selection ran on the instances reflection read, and a seed the
  metric scores identically on every validation instance reports that there was
  nothing to rank. Pass `validationSet: "reuseTraining"` to accept the reuse by
  name. Custom optimizers implementing `OptimizerResult` or emitting `RunFinished`
  must now populate `warnings`.

### Patch Changes

- 3f5824b: The long-form guides ship in the tarball, under `docs/`, so an installed copy
  documents the version installed rather than whatever `main` has become. Two of
  them are new: `data-prep.md` on splitting a dataset a search can be trusted
  with, and `metric-preflight.md` on checking a metric separates candidates —
  and moves over a wide enough interval — before a budget is spent on it.

  Doc comments on the API carry the traps that belong beside the code: that
  `weight: 0` removes a criterion from the aggregate but not from
  `objectiveScores`, that a `gate` and a heavy `weight` on the same criterion
  enforce a requirement twice and narrow the range a search has left to move in,
  and that a SIMBA run has to be funded past its finalist reserve before any step
  happens.

## 0.1.0

### Minor Changes

- ca8a541: A harvested rollout cannot end the demo block it is stored in.

  Demo blocks are delimited by `<demo>`, and the values inside them were written
  raw. A system that quotes its own prompt back produces the one output that
  breaks: a rollout worth keeping whose text carries `</demo>`, which closes the
  block early and leaves the rest of it as loose text. `parseDemos` then returned
  a demo that was not the one stored, and SIMBA reparses and rewrites its demo
  components at every step, so the loss compounded over a run instead of showing
  up once.

  `<demo>`, `<input>` and `<output>` are now escaped in serialized demo values and
  unescaped on the way back, so a demo containing a demo round trips as itself.
  The escape is escaped first, so a value that already reads `&lt;demo>` survives
  too. Demo blocks written by earlier versions still parse; blocks whose values
  contain those tags will render them escaped from now on, which changes the text
  a component holds and so the candidates a run compares.

  A custom `renderDemo` is responsible for its own escaping: the library cannot
  know which part of what a renderer emits is the delimiter it meant to write.

- b25abd2: SIMBA's advice proposer sees what each component already says.

  `buildAdvicePrompt` named the components it wanted advice for but never showed
  their text, while the advice it produces is appended to that text rather than
  replacing it. A proposer that cannot read what it is appending to writes blind:
  it restates guidance the component already carries, and it cannot correct
  guidance that is wrong, since contradicting a line it never saw is not something
  it can choose to do. SIMBA's reference implementation passes the current
  instructions for exactly this reason.

  `AdvicePromptArgs` now carries `current`, a map from component name to what that
  component holds, and the prompt renders each as a `<component name="…">…
</component>` block followed by the instruction not to restate advice already
  present. A custom `AdvicePromptBuilder` receives the extra field and may ignore
  it; anything constructing `AdvicePromptArgs` by hand, or parsing the built
  prompt's component list, has to be updated.

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

- b25abd2: A demonstration lands in a component without erasing what else it says.

  SIMBA's `appendDemo` rebuilt a demo component from its demos alone, so any text
  the component held that was not a demo block was gone the first time a rollout
  was harvested into it — including the advice `appendRule` had just written
  there. The two mutations could not share a component, which is why
  `instructionComponents` defaulted to the components `demoComponents` did not
  name, and why a candidate with a single component got one mutation instead of
  two. SIMBA's reference implementation appends demos and instructions to the same
  predictor; a component is the closest thing this library has to one.

  `replaceDemos` rewrites the demo blocks in a text and leaves the rest of it
  alone, and both `appendDemo` and the loop's demo-dropping now go through it. A
  component named in `demoComponents` can hold instructions too, and the default
  `instructionComponents` falls back to every component when every component holds
  demos, rather than leaving `appendRule` with nowhere to write and throwing.

  Runs where demo and instruction components were already disjoint are unaffected
  except that a demo block now keeps its position in the text rather than
  replacing it. Runs where they overlapped were losing text and are not
  comparable to their old results.

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

- 2606e36: Rollout accounting survives a resume, and the numbers a run reports name the
  candidate it returns.

  `maxCostUsd` is a ceiling on the run rather than on the segment: every snapshot
  now carries the usage already spent, and a resumed run folds it back in instead
  of restarting its token and dollar totals at zero. Harvesting is part of that
  total — `harvestRollouts` and `harvestFewShotExamples` report the `usage` their
  own evaluator spent, and MIPRO and bootstrapped few-shot search absorb it
  through `Evaluator.absorbUsage`, so the pass that collects demos is no longer
  invisible to a cost ceiling that is supposed to bound it.

  SIMBA keys a minibatch by dataset row instead of by position within the batch.
  Two steps drawing different rows shared the ids `0..n-1`, so a candidate that
  recurred across steps could be served a cached score another instance had
  measured. Its default `instanceId` hashes the datum, as bootstrapped few-shot
  search's now does and as the other four optimizers already did. This is the
  breaking half of the release: a seeded run that changed nothing now reaches
  different candidates, because a score measured on another row no longer decides
  one. `bench/results/latest.json` moves with it.

  OPRO sweeps the held-out set with the candidate it returns. A run that screens
  on a `scoringSetSize` subset can end on an incumbent the closing full sweep
  never confirmed, and `testScore` described that unreturned candidate rather than
  `bestCandidate`.

  Candidate ids continue across a resume in OPRO, MIPRO, bootstrapped few-shot
  search and random search. Reporters key rows by `candidateId`, and a counter
  that restarted at zero made a resumed run overwrite the run it continued.

  `SAMPLING_POOL` is exported from `textopt/testing`, which the README had
  documented but the module did not export.

- b2be4d6: A comparison reports what each entrant was given, and claims only what its
  seeds actually put to the test.

  `compare()` reports the work a run got for free. Two entrants held to the same
  `maxMetricCalls` do not do the same amount of work: one that revisits candidates
  it has already scored is served from the cache, and one that never revisits pays
  for every rollout, so ranking them on `metricCalls` alone credits the difference
  to the search. Every run now carries `cacheHits` and `reflectionCalls`, summarized
  as `meanCacheHits` and `meanReflectionCalls`, and reflection calls are counted
  because no metric budget covers them — an entrant can be cheap on rollouts and
  expensive on proposals. `OptimizerResult` carries both: `cacheHits` on every
  optimizer, `reflectionCalls` on the five that propose text. Anything implementing
  that interface by hand has to report `cacheHits`.

  A p-value is withheld where the seeds never earned one. `compare()` reported
  `pValueVsWinner` for every non-winner, including entrants whose margin over the
  winner was identical at every seed — a deterministic search against a
  deterministic model, or one whose seed reaches nothing that varies. A sign-flip
  test over n seeds answers a question about n independent trials; n copies of one
  realization is a single trial reported as n, and the p-value it produces goes as
  low as 2^-n while resting on nothing. Those comparisons now report no p-value at
  all, and `distinctScores` says how many distinct outcomes an entrant actually
  had, so a row of identical scores is visible rather than inferred. A margin of
  exactly zero at every seed is unaffected: p = 1 claims nothing and is honest.

  `pValueVsWinnerHolm` corrects for the family the raw p-value is read against.
  Comparing six entrants against a winner runs five tests, and the smallest of five
  is smaller than one test's worth of evidence. Holm-Bonferroni step-down adjusts
  each surviving comparison against the whole family — including the slots held by
  comparisons withheld above, which are still members of it.

  Exact enumeration now reaches twenty seeds rather than sixteen. `signFlipPValue`
  builds the reachable sums by doubling instead of re-summing each of the 2^n sign
  masks, which costs O(2^n) rather than O(2^n · n) — about four times faster at
  n = 20, 16ms against 63ms. Comparisons between sixteen and twenty seeds get the
  exact p-value where they previously fell back to the normal approximation.

  Random search starts a round whenever one more sweep is affordable. It required
  the whole round up front — every variant proposed and scored — which stranded up
  to `variants * |validationSet| - 1` rollouts unspent at the end of a run, around
  a fifth of a typical budget at the default settings. A variant needs only its own
  sweep to be compared against the incumbent, and the round already truncates to
  what the remainder can fund whole sweeps for. The cost is bounded and paid once:
  the final round proposes up to `variants - 1` texts it never scores. Runs at a
  fixed `maxMetricCalls` will evaluate more variants than before and can select a
  different candidate.

### Patch Changes

- 3336175: `harvestRollouts` runs a candidate over data and returns the rollouts the
  metric rewarded, without the demo ceiling `harvestFewShotExamples` applies, and
  `toTrainingJsonl` serializes them as chat-messages JSONL for distilling a run
  into a smaller model. `harvestFewShotExamples` keeps its behaviour and now
  calls the same primitive; [Distilling a run](docs/distillation.md) covers which
  data to sweep and how much of the optimized candidate to leave in the training
  input.
- 7809e1d: SIMBA and bootstrapped few-shot search take a `concurrency` option, and OPRO
  and random search now apply theirs to evaluation as well as to proposals: a
  round's screens or sweeps run together, SIMBA overlaps the candidates a step
  built and the finalist sweeps, and a bootstrap candidate's sweep overlaps the
  harvest behind it. Every fan-out draws its random stream and prices its
  schedule before dispatching, and commits in the order the search proposed, so a
  seeded run reaches the same candidates and spends the same rollouts at any
  concurrency. Defaults stay at one evaluation at a time.
