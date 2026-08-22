# Tuning a run

What a search spends and how to bound it: the arithmetic for pricing a run
before starting it, what to read when one disappoints, the guards for a metric
that does not return the same number twice, and the ceilings on money, time,
and repeated work.

## Sizing a run

`maxMetricCalls` is the only hard bound on a search, and a run that cannot afford its next unit of work stops rather than throws: the result carries `stopReason: "budgetExhausted"` and whatever had been found by then. An underfunded run looks exactly like a finished one, so price it before starting it.

GEPA, MIPRO, OPRO, and random search sweep the seed candidate over the validation set before anything else, and bootstrap search's first candidate is the zero-shot one, which is that same sweep. SIMBA scores its seed alongside its finalists instead. After that, each spends in units of its own — `|val|` below is the size of the validation set, `|train|` the training set:

| Optimizer                  | One unit of search                                                                   | Charged besides                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `GepaOptimizer`            | an iteration: `perIteration × minibatchSize × 2 + \|val\|`                           | —                                                                                                       |
| `SimbaOptimizer`           | a step: `(candidates + 1) × minibatchSize`                                           | `min(candidates + 1, maxSteps + 1) × \|val\|`, reserved before the first step                           |
| `MiproOptimizer`           | a trial: `minibatchSize`                                                             | `\|val\|` every `fullEvalInterval` trials, and up to `demoSets × \|train\|` to bootstrap each demo menu |
| `OproOptimizer`            | a round: up to `proposalsPerRound × \|val\|`, or `× scoringSetSize` when that is set | `\|val\|` every `fullEvalInterval` rounds, once `scoringSetSize` is set                                 |
| `BootstrapSearchOptimizer` | a candidate: `\|val\|`, plus up to `\|train\|` to harvest its demos                  | —                                                                                                       |
| `RandomSearchOptimizer`    | a round: `variants × \|val\|`                                                        | —                                                                                                       |

GEPA's doubling is the parent: each proposal scores its parent and its child on the same minibatch, because acceptance is a paired comparison rather than a threshold. The trailing `|val|` is the sweep a child earns by improving, reserved before the iteration starts rather than discovered missing once there is something to promote. The others refuse work for the same reason — MIPRO stops as soon as it can no longer afford a sweep, because a reading nothing can act on buys nothing.

SIMBA is the one worth doing the arithmetic for, because its reserve comes off the top. Under its defaults — `candidates: 6`, `minibatchSize: 32`, `maxSteps: 8` — against a 50-instance validation set, it holds back 350 rollouts for the finalists and spends 224 per step, so eight steps need about 2,150. The same run under `maxMetricCalls: 600` takes one step and stops.

### Sizing the sets

`validationSet` defaults to `trainingSet`. That is the right default for a first run and the wrong number to report: the search selected against those instances, so `bestScore` is fitted to them. Under reflective search it is worse than ordinary overfitting — the reflection prompt asks the model to mine domain facts out of the traces it is shown, so those facts come out of the very instances that then select the candidate carrying them.

A run that took the default says so: `result.warnings` carries a `validationSetReusesTraining` entry, and so does the `finish` event. Pass `validationSet: "reuseTraining"` to accept the reuse by name and silence it. [Held-out evaluation](./evaluation.md#held-out-evaluation) is how to find out what it cost.

Validation size multiplies almost every row of the table, so it decides what a run costs. Shrinking it is the wrong lever — it makes the number that picks the winner noisier. Screen on something smaller and sweep rarely instead: that is what OPRO's `scoringSetSize`, MIPRO's minibatch trials, and GEPA's minibatch screening are for.

Minibatch defaults differ by an order of magnitude between optimizers — GEPA 3, SIMBA 32, MIPRO 35 — and do not transfer. GEPA compares a child against its own parent on the same instances, so three of them already say something. MIPRO hands the batch mean to its surrogate as an absolute reading of a configuration. SIMBA ranks the instances in a batch by how much its programs disagreed on them. Carry one optimizer's number to another and the search reads noise.

### When a run disappoints

| What happened                                                             | Where to look                                                                                                                  |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| stopped short of `maxIterations`, `maxSteps`, `maxTrials`, or `maxRounds` | `stopReason`. `budgetExhausted` is the arithmetic above, not a failure                                                         |
| `stopReason` is `reflectionBudgetExhausted`                               | GEPA's `reflection.maxCalls` or OPRO's `maxReflectionCalls`. Both are separate from the rollout budget and default unbounded   |
| `bestScore` improved but `testScore` did not                              | the validation set is too small or too easy to separate candidates. The gap is the measurement working                         |
| accepted candidates do not hold up when re-evaluated                      | metric noise — [Noisy metrics](#noisy-metrics), including the cost of turning both guards on                                   |
| proposals repeat themselves                                               | `reflection.strategies` for the framing, `rejectedProposalMemory` for what the prompt is told has already failed               |
| MIPRO settles on the seed                                                 | the menus were the search space: read `result.menu`, add `componentOptions`, and set `multivariate: false` when trials are few |
| the run cost more money or took longer than expected                      | `maxCostUsd` and `maxWallClockMs` — [Budgets, cost, and time](#budgets-cost-and-time)                                          |

### What to try first

`BootstrapSearchOptimizer` answers the cheapest question worth asking first — whether the instruction is already fine and consistency is what is failing — and it calls no proposal model to do it, so a run costs rollouts and nothing else. Its two answers are both worth having: on the benchmark's `demonstrated` task it beats every entrant that searches instructions except the two GEPA rows, and on `clean` it scores zero, which is what "your system is not inconsistent, it is mis-instructed" looks like. Reach for a reflective search once that has been ruled out, and pick between them with [`compare()`](./evaluation.md#comparing-optimizers) under one budget rather than from the [benchmark table](./benchmark.md).

## Minibatch sizes do not transfer

The defaults differ by an order of magnitude — GEPA 3, SIMBA 32, MIPRO 35 — because they mean different things. GEPA compares a child against its own parent on the same instances, so three already say something. MIPRO hands the batch mean to a surrogate as an absolute reading. SIMBA ranks instances within a batch by how much its programs disagreed. Carrying one number to another optimizer makes it read noise.

What three instances buy in GEPA is a paired comparison, not a filter. On a metric that moves smoothly — a rubric, a judge, anything averaging several partial credits — almost every proposal that is any good at all wins that comparison, and a run accepting ten of ten proposals at the gate is GEPA behaving as published rather than a gate that has broken. The screening that matters then happens on the full sweep, which is what decides whether the incumbent actually moves. Read `candidateAccepted` for that, not the acceptance rate.

If you want the gate itself to reject more, `minibatchSize` is the lever: it is the only thing that makes the paired comparison harder to win. It is not free — every proposal costs `minibatchSize × 2` rollouts, its parent's and its own — so raising it trades iterations for screening. Widen it when a run promotes freely and the promotions do not hold up on the sweep.

## Noisy metrics

The default acceptance rule takes any minibatch improvement, and the default winner is the highest validation mean. Both are the right reading when a rollout of the same text on the same instance always scores the same. When it does not — a sampled model, a judge, a flaky tool — a run accumulates changes that only ever won a coin flip.

```ts
import {
  lowerBoundEvaluationPolicy,
  pairedPermutationAcceptance,
} from "textopt/gepa";

new GepaOptimizer({
  minibatchSize: 8,
  acceptance: pairedPermutationAcceptance({ alpha: 0.2 }),
}).optimize({
  ...task,
  valEvaluationPolicy: lowerBoundEvaluationPolicy({ z: 1 }),
});
```

`pairedPermutationAcceptance` accepts only when the paired per-instance improvement survives a sign-flip test at `alpha`. `lowerBoundEvaluationPolicy` returns the candidate with the best mean minus `z` standard errors, rather than the best mean.

Both are strictly more conservative. On the benchmark that buys a little where it should and costs nothing where it should not: on the `noisy` task the pair scores 0.931 against plain GEPA's 0.920, and on the noiseless `clean` task 0.945 against 0.947 — a difference no larger than the seed spread. Neither gap clears significance over twenty seeds (Holm-adjusted p = 0.088 and 0.334), so read this as "not expensive" rather than as a demonstration that it works. Where it does lose clearly is the pipeline task, 0.835 against 0.891 at p = 0.027: a significance bar on every acceptance is expensive when improvements to one component only pay off after another is finished. Turn them on when you have measured the metric's own variance and found it large, not on principle.

A minibatch also has to be wide enough for the test to say anything: a sign-flip test over three instances cannot produce a p-value below 0.125, so at the default `minibatchSize` of 3 no proposal can ever be accepted at `alpha` below that. `GepaOptimizer` throws on that combination at construction. Both halves are reasonable on their own and only their product is wrong, so the constructor is the one place that can see it — and the run it prevents is one that spends its entire budget, returns the seed, and reports a `stopReason` that looks like any other.

## Budgets, cost, and time

`maxMetricCalls` bounds rollouts. It does not bound money or time, and on a long run neither of those follows from it:

```ts
await optimizer.optimize({
  ...task,
  maxMetricCalls: 2000,
  maxCostUsd: 25, // stops at the first decision point past the ceiling
  maxWallClockMs: 30 * 60_000, // overruns by at most one evaluation
  retry: { attempts: 2, delayMs: 500 },
});

result.usage; // { inputTokens, outputTokens, totalTokens, costUsd, rollouts }
result.testUsage; // the same, for the held-out sweep, which no ceiling bounds
```

- **`maxCostUsd`** exists because reflective search grows the text it optimizes, so late rollouts cost more than early ones. It is checked between evaluations and reads whatever usage the adapter reported; an adapter that reports none can never trigger it. `priceUsage` fills in `costUsd` on a rollout's usage from a `TokenPricing` table; adapters call it so there is something to read.
- **`maxWallClockMs`** exists because a run behind a rate limit spends almost nothing and takes as long as the provider makes it take. This is what makes an optimizer safe to put behind a request timeout or a nightly job. `stopReason` is `"deadlineReached"`. Both ceilings are checked between evaluations, so a run overruns them by whatever it had in flight when they were reached — one evaluation at the default concurrency, and up to `concurrency` of them above it.
- **Neither ceiling bounds the held-out sweep.** A `testSet` is measured once, after the search has already stopped, and charging it would let the size of a held-out set decide which candidate wins. So it is reported apart from the search: `testMetricCalls` rollouts costing `testUsage`, neither of them inside `metricCalls` or `usage`. Budget for it separately — a `testSet` the size of the validation set costs one full sweep on top of whatever `maxCostUsd` allowed.
- **`retry`** re-runs instances the adapter marked `transient`. A rate limit or a 5xx otherwise costs the instance either an unexplained zero or a hole in the candidate's coverage. Retries are charged like any other rollout but never overdraw the budget. Defaults to two attempts, 500 ms apart, doubling.

## Concurrency

Every optimizer runs one evaluation at a time by default. `concurrency` raises that, and each optimizer applies it to the part of its loop where the work is genuinely independent:

| Optimizer             | What overlaps                                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| GEPA                  | `proposals.concurrency` — the proposals of an iteration, and then the validation sweeps of the children that survived screening |
| SIMBA                 | the candidates a step built, scored on its minibatch, and the finalist sweeps that pick the winner                              |
| OPRO                  | the proposals of a round, and the screen each of them is scored on                                                              |
| Random search         | the variants of a round, and the sweep each of them is scored on                                                                |
| Bootstrapped few-shot | a candidate's sweep with the harvest of the candidates behind it                                                                |
| MIPRO                 | the instruction proposals for one component                                                                                     |

Raising it does not change what a run finds. Every one of these fans out only after the random stream has been drawn and the whole schedule has been priced against the allowance, and commits results in the order the search proposed them rather than the order they returned — so a seeded run reaches the same candidates, the same winner and the same rollout count at any concurrency. The parts that would not survive that are left in sequence: MIPRO's trials condition on the observations before them, and SIMBA's trajectory samples and mutations read state the sample before them wrote.

What does change is what the provider sees. The limits multiply — `compare` concurrency, then the optimizer's, then the adapter's — so GEPA at `proposals: { concurrency: 4 }` over an adapter at `concurrency: 4` is sixteen calls in flight. Past what a provider tolerates this stops being a timing question: a throttled rollout comes back marked `transient`, which is left out of the candidate's mean rather than scored as a zero, retried at the run's expense, and never cached. A run that spends its allowance on retries and measures its candidates on fewer instances than it asked for looks, from the outside, like a seed that went badly.

Two smaller costs come with raising it. The cost and deadline ceilings overrun by whatever was in flight when they were hit. And bootstrapped few-shot search checkpoints once per wave rather than once per candidate, so a killed run loses up to `concurrency` candidates instead of one; it also ignores `concurrency` when `stopAtScore` is set, since a wave cannot know it has already passed the target.

## Caching, checkpoints, resume

- **Caching.** Cache keys include the split, complete candidate, and instance ID. Cache hits do not count against the metric budget. Instance IDs default to a content hash, falling back to the row position for a datum whose whole content the hash cannot read — one that will not serialize, and one that JSON does not reach at all, such as a bare Map, a Set, or a class instance holding its state privately. The fallback reads the whole datum, not its fields: two rows that differ only inside a nested Map, Set, or function-valued property serialize alike, hash alike, and share cached scores. Pass `instanceId` for data like that, and for non-serializable data or readable trace IDs. Set `cache: false` to disable caching.
- **`cacheNamespace`.** A cached score is a measurement of a whole system, not of a candidate. Set `cacheNamespace` to name the model id, decoding settings, and scorer version, and change it whenever anything outside the candidate text changes — otherwise a run silently reuses scores measured under a system it is no longer running.
- **Durable caching.** `createFileCache({ path, namespace })` from `textopt/file-cache` is an append-only log that outlives the process, so a crashed run, a re-run with a changed budget, and a second experiment over the same validation set do not pay for identical rollouts again. `namespace` is required there rather than optional as `cacheNamespace` is: a log on disk outlives every part of the measured system a key does not name, and entries written under one namespace are never served to a run under another. It needs `node:fs`; for Redis or SQLite, implement `EvaluationCache` yourself.
- **Checkpoints.** Every optimizer takes `onCheckpoint` and `resumeFrom`, and returns its final `snapshot`. Each restores exactly the state that is expensive or unrepeatable: GEPA's candidate pool, rejections and merge state; MIPRO's option menus and surrogate observations; OPRO's screening slice and score histories; SIMBA's program pool and step winners; the search's budget, RNG, sampler position and cached scores throughout. A fingerprint refuses a checkpoint from a different seed candidate, instance set, or random seed. Snapshots are plain JSON and are never mutated by the run that resumes from them.
- **Events.** Every optimizer takes `reporters`, an array of `Reporter<Event>` with an optional `onEvent` and an optional `flush`. `consoleReporter()` from the root and `jsonlReporter({ path })` from `textopt/file-reporter` cover a first run: one line per event on the terminal, and the same events as JSON Lines on disk. Each search emits its own discriminated union, and all of them emit `start`, `evaluation`, `rollout`, `candidateAccepted` and `finish` with a shared payload — so a reporter that narrows with `isCandidateAccepted` and its siblings drops into any optimizer without knowing which one it is.
- **Writing one.** Use `createReporter({ on: { … } })` and parameterize it with the optimizer's own event type. A reporter annotated `Reporter<{ type: string }>` accepts any tag, so a handler for an event that does not exist compiles, runs, and prints nothing for the length of the run. `createReporter` makes that a compile error, and a handler for an event this particular optimizer does not emit is warned about when the run starts.
- **Progress.** `rollout` is the only event that moves during a validation sweep. It comes from the adapter, so an adapter of your own has to call the `onRollout` it is handed — see [Adapters](./adapters.md#the-adapter).
- **Result.** Every result carries `bestCandidate`, `bestScore`, `metricCalls`, `usage`, `stopReason`, and `snapshot`, plus whatever its own search can report.
