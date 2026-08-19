# textopt

Core interfaces and optimizers for textopt.

This package has no runtime dependencies. For an overview of the algorithms and guidance on choosing one, see the [project README](https://github.com/ctdio/textopt#readme).

## Entry points

| Import                     | Contains                                                                                                                            |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `textopt`                  | Shared contracts, evaluator, judge, `compare()`, demo utilities, cache, and concurrency helper. Optimizer classes are not exported. |
| `textopt/gepa`             | `GepaOptimizer`, GEPA types, events, checkpoints, the pipeline adapter, and configurable strategies.                                |
| `textopt/simba`            | `SimbaOptimizer`, its advice prompt, and its bucket-ranking helpers.                                                                |
| `textopt/opro`             | `OproOptimizer` and its types and events.                                                                                           |
| `textopt/mipro`            | `MiproOptimizer`, its types, and the standalone `proposeConfiguration` TPE function.                                                |
| `textopt/bootstrap-search` | `BootstrapSearchOptimizer` and its types.                                                                                           |
| `textopt/random-search`    | `RandomSearchOptimizer` and its types.                                                                                              |
| `textopt/file-cache`       | `createFileCache`, an append-only durable `EvaluationCache`. The only entry point that uses `node:fs`.                              |
| `textopt/testing`          | Deterministic fixtures for testing optimizers and adapters without an LLM.                                                          |

## `textopt`

```ts
import {
  assertResumable,
  bootstrapDemos,
  buildJudgePrompt,
  compare,
  componentNames,
  createDeadline,
  createEvaluator,
  createJudge,
  createMemoryCache,
  formatDemos,
  mapWithConcurrency,
  parseDemos,
  parseProposedText,
  priceUsage,
  runFingerprint,
} from "textopt";
import type {
  Adapter,
  BatchSampler,
  BootstrapResult,
  Candidate,
  CachedScore,
  Comparison,
  ComparisonRun,
  ComparisonSummary,
  Deadline,
  Demo,
  DemoRenderer,
  EvaluationEvent,
  Evaluator,
  ScoredBatch,
  EvaluateArgs,
  EvaluationBatch,
  EvaluationCache,
  EvaluationContext,
  EvaluationPhase,
  EvaluationSplit,
  Judge,
  JudgeCriterion,
  JudgePromptBuilder,
  Optimizer,
  OptimizerResult,
  OptimizerTask,
  RetryPolicy,
  Rng,
  RolloutUsage,
  ScoreResult,
  TextModel,
  TokenPricing,
  UsageTotals,
} from "textopt";
```

### Contracts

**`Candidate<K extends string>`** is a `Record<K, string>` containing the text components to optimize. `K` is inferred from the seed candidate, so misspelled component names fail type checking.

**`Adapter<Datum, Trajectory, Output, K>`** connects an optimizer to the evaluated system:

```ts
evaluate(args: EvaluateArgs<Datum, K>): Promise<EvaluationBatch<Trajectory, Output>> | EvaluationBatch<Trajectory, Output>
```

`EvaluateArgs` contains the `batch`, `candidate`, `captureTraces`, an optional abort `signal`, and a `run` context with `iteration`, `phase`, `split`, and `candidateId`. The run context can be forwarded to a tracing system.

`EvaluationBatch` contains one output and score per instance. Higher scores are better. It may also include `feedback`, `trajectories`, `objectiveScores`, and `transient` flags. Reflective optimizers use `feedback` to generate revisions.

**`ScoreResult`** is the common return type for per-instance scorers: `score`, with optional `feedback`, `objectiveScores`, and `transient`.

**`transient`** marks scores caused by infrastructure failures such as rate limits, 5xx responses, or network errors. Transient scores are not cached.

**`Optimizer<Stop extends string>`** defines `optimize(task: OptimizerTask) => Promise<OptimizerResult>`. `OptimizerTask` contains the shared run inputs: `seedCandidate`, `trainingSet`, `validationSet`, `testSet`, `adapter`, `maxMetricCalls`, `maxCostUsd`, `maxWallClockMs`, `cacheNamespace`, `retry`, and `signal`. `OptimizerResult` contains `bestCandidate`, `bestScore`, `bestOutputs`, `metricCalls`, `usage`, `testScore`, `testMetricCalls`, and `stopReason`. Optimizer-specific task and result types extend these interfaces.

**`maxCostUsd`** and **`maxWallClockMs`** are checked between evaluations, so a run overruns by at most one of them. Neither follows from `maxMetricCalls`: reflective search grows the text it optimizes, so late rollouts cost more than early ones, and a run behind a rate limit spends almost nothing while taking as long as the provider makes it take.

**`cacheNamespace`** scopes every cache key to the system the rollouts were measured under — model id, decoding settings, scorer version. Change it whenever anything outside the candidate text changes.

**`retry`** is a `RetryPolicy` of `{ attempts = 2, delayMs = 500 }`. Instances the adapter marked `transient` are re-run, with the delay doubling per attempt. Retries are charged like any other rollout and never overdraw the budget.

**`UsageTotals`** (`inputTokens`, `outputTokens`, `totalTokens`, `costUsd`, `rollouts`) is summed from the `RolloutUsage` entries an adapter reports. Zero throughout when the adapter reports none.

**`testSet`** is excluded from search and evaluated once against the winner. Because candidates are selected on `validationSet`, `bestScore` may be fitted to it. `testScore` measures held-out performance. Test rollouts are reported as `testMetricCalls` and do not count against `maxMetricCalls`.

**`TextModel`** is the provider-independent interface `({ prompt, signal }) => Promise<string>`.

### Values

**`createMemoryCache({ maxEntries = 100_000, entries })`** returns an in-memory `EvaluationCache`. Pass `entries` to restore checkpointed scores. When full, the cache evicts the oldest entry.

For Redis, SQLite, or file-backed caching, implement **`EvaluationCache`** with `get(key)`, `set(key, cached)`, and optional `entries()`. A durable store does not need `entries()`.

**`mapWithConcurrency({ items, limit, task, signal })`** maps items in order while limiting concurrent tasks. Running tasks settle before an error is propagated.

**`componentNames(candidate)`** returns `Object.keys(candidate)` while preserving the component-name union.

**`createEvaluator({ adapter, budget, cache, cacheNamespace, retry, trackOutputs, onEvaluation, signal, cacheHits })`** handles adapter calls, caching, budget accounting, transient scores, and evaluation events. `evaluate` returns a `ScoredBatch`. `evaluateTraced` returns an `EvaluationBatch`, or `null` when the remaining budget cannot cover the batch. A batch that exceeds the charged budget throws `BudgetExhausted`. All included optimizers use this evaluator.

**`bootstrapDemos({ adapter, candidate, trainingSet, minScore, maxDemos, batchSize, maxMetricCalls, rng, renderDemo, signal })`** evaluates a candidate on `trainingSet` and keeps the rollouts the metric rewarded. Omit `minScore` to keep any rollout scoring above zero, as MIPROv2 does without a `metric_threshold`; pass a number to require at least that score. It returns the selected `demos`, a formatted `block`, and the metric calls used. It does not use the score cache because it needs rollout outputs.

**`formatDemos(demos, { render })`** and **`parseDemos(text)`** write and read the `<demo>`, `<input>`, and `<output>` block format.

**`parseProposedText(text)`** extracts a proposal from a reflection response, including responses with fenced blocks or surrounding commentary.

**`createJudge({ model, criteria, scale = 5, renderInput, renderOutput, buildPrompt })`** returns a `Judge<Datum, Output>`: `({ input, output, expected, signal }) => Promise<ScoreResult>`. Each `JudgeCriterion` is graded on a small integer scale and normalized; per-criterion values are returned as `objectiveScores` and the aggregate `score` is their mean. A criterion the judge failed to grade returns a transient score rather than a zero, so the instance is retried instead of recorded as a failure. **`buildJudgePrompt`** is the default template and implements `JudgePromptBuilder`.

**`compare({ entrants, seeds, concurrency = 1 })`** runs each entrant over every seed and returns a `Comparison` of `winner`, `summaries`, and `runs`. Entrants are `({ seed }) => Promise<OptimizerResult>`, so the caller builds the optimizer-specific task. Ranking is on `testScore` where a run reports one, because the validation score is the number the search selected against. Each `ComparisonSummary` carries `meanScore`, `sdScore`, `minScore`, `maxScore`, `meanMetricCalls`, `meanCostUsd`, and a paired sign-flip `pValueVsWinner`.

**`priceUsage({ usage, pricing })`** fills in `costUsd` on one rollout's `RolloutUsage` from a `TokenPricing` table. Adapters call it so a run's `usage.costUsd` and its `maxCostUsd` ceiling have something to read; without it both stay zero.

**`createDeadline({ maxWallClockMs, now })`** returns a `Deadline` with `exceeded()` and `remainingMs()`. The optimizers build one from `maxWallClockMs`; `now` is injectable so a deadline can be tested without waiting.

**`runFingerprint({ seedCandidate, trainingIds, validationIds, seed, cacheNamespace })`** and **`assertResumable({ fingerprint, snapshot })`** are the shared checkpoint guard. Every optimizer stamps its snapshot with a fingerprint and refuses one that does not match, rather than silently scoring old candidates against new data. **`candidateFingerprint(candidate)`** hashes a candidate's text.

**`BatchSampler<Datum>`** and **`Rng`** are type-only exports. Their default implementations are internal.

## `textopt/gepa`

```ts
import { GepaOptimizer } from "textopt/gepa";

const gepa = new GepaOptimizer(config); // GepaConfig: stateless, reusable
const result = await gepa.optimize(task); // GepaTask: one problem
```

Optimizer instances do not retain state between runs.

### `GepaConfig`

These options control search behavior and can be reused across runs.

| Option                     | Default                   | Effect                                                                                                        |
| -------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `minibatchSize`            | `3`                       | Instances per screening batch.                                                                                |
| `maxIterations`            | `Infinity`                | Iteration ceiling. The metric budget usually binds first.                                                     |
| `seed`                     | `0`                       | Seeds the run's random stream.                                                                                |
| `candidateSelector`        | `paretoSelector()`        | Which candidate becomes the next parent.                                                                      |
| `acceptance`               | `improvementAcceptance()` | Whether a screened child beats its parent.                                                                    |
| `merge.enabled`            | components > 1            | System-aware merge of two lineages.                                                                           |
| `merge.maxInvocations`     | `5`                       | Ceiling on merges attempted per run.                                                                          |
| `merge.valOverlapFloor`    | `5`                       | Validation instances two lineages must share to be eligible. A validationSet smaller than this never merges.  |
| `skipPerfectScore`         | `true`                    | Skip reflection when the parent already scores perfectly on the minibatch.                                    |
| `perfectScore`             | `1`                       | The per-instance score treated as leaving no room.                                                            |
| `rejectedProposalMemory`   | `3`                       | Rejected texts per component included in reflection. `0` disables it. This behavior is not in the GEPA paper. |
| `proposals.perIteration`   | `1`                       | Mutations drawn per iteration, each with its own parent and minibatch.                                        |
| `proposals.concurrency`    | `1`                       | How many may be in flight at once.                                                                            |
| `proposals.selection`      | `"all"`                   | Which improving proposals to keep: `"all"`, `"best"`, or `{ keep: n }`.                                       |
| `reflection.maxCalls`      | unbounded                 | Hard ceiling on reflection calls. The run stops when it is reached.                                           |
| `reflection.maxRecords`    | unbounded                 | Records shown per component. The worst-scoring ones are kept.                                                 |
| `reflection.maxCharacters` | unbounded                 | Rough ceiling on the serialized records.                                                                      |
| `reflection.buildPrompt`   | `buildReflectionPrompt`   | Replaces the prompt template. Custom proposers ignore it.                                                     |
| `reflection.strategies`    | none                      | Rotates over several prompt templates, one per proposal slot. Mutually exclusive with `buildPrompt`.          |
| `checkpointCache`          | `true`                    | Include cached scores in every snapshot.                                                                      |
| `trackBestOutputs`         | `false`                   | Keep validation outputs so the winner's can be read back.                                                     |
| `raiseOnError`             | `true`                    | Rethrow adapter failures instead of skipping the iteration.                                                   |

**What a run costs.** One validation sweep for the seed, then per iteration `proposals.perIteration × minibatchSize × 2` — every proposal screens its parent and its child on the same minibatch — plus `validationSet.length` for the sweep a surviving child earns — less than that where `valEvaluationPolicy` subsamples. All of it is reserved before the iteration begins, so a run never starts an iteration it cannot afford to promote out of.

### `GepaTask`

`GepaTask` extends `OptimizerTask` with GEPA-specific run inputs.

Required: `seedCandidate`, `trainingSet`, `adapter` (a `GepaAdapter`), `reflect` (a `TextModel`), and `maxMetricCalls`.

| Option                | Default                                                                              |
| --------------------- | ------------------------------------------------------------------------------------ |
| `validationSet`       | the trainingSet                                                                      |
| `testSet`             | none. Held out of the search and scored once, on the winner                          |
| `componentSelector`   | `roundRobinComponentSelector()`                                                      |
| `batchSampler`        | an epoch-shuffled sampler over `minibatchSize`                                       |
| `valEvaluationPolicy` | `fullEvaluationPolicy()`                                                             |
| `instanceId`          | a content hash of the datum, falling back to its position when it will not serialize |
| `cache`               | a per-run memory cache. Pass `false` to disable                                      |

`onEvent`, `onCheckpoint`, `resumeFrom`, and `signal` have no defaults.

TypeScript infers component names and the datum type from `seedCandidate` and `trainingSet`. Other fields use `NoInfer` and are checked against those inferred types.

### `GepaAdapter`

Extends `Adapter` with what reflection needs:

```ts
makeReflectiveDataset(args: MakeReflectiveDatasetArgs<Datum, Trajectory, Output, K>): ReflectiveDataset<K>
proposeNewTexts?(args: ProposeArgs<K>): ComponentPatch<K>   // optional
```

Both methods may be synchronous or asynchronous. `ReflectiveDataset` is a partial map from component names to `ReflectiveRecord[]`. Each record contains `inputs`, `generatedOutputs`, `feedback`, `score`, and a typed `evidence` field. Adapters only need to return records for the requested components.

When `proposeNewTexts` is implemented, the adapter generates proposals without calling `reflect`. The task type still requires `reflect`, so offline runs can pass a stub.

### Strategies

| Strategy                      | Arguments                                | What it does                                                                                  |
| ----------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| `paretoSelector`              | `{ epsilon = 0, frontier = "instance" }` | Samples a parent from the frontier. `frontier` is `"instance"`, `"objective"`, or `"hybrid"`. |
| `currentBestSelector`         | none                                     | Always picks the highest aggregate score.                                                     |
| `epsilonGreedySelector`       | `{ epsilon }`                            | The best candidate, with a uniform random pick at rate `epsilon`.                             |
| `topKParetoSelector`          | `{ k, epsilon = 0 }`                     | Pareto selection restricted to the top `k` by aggregate score.                                |
| `roundRobinComponentSelector` | none                                     | One component per iteration, walking a per-candidate cursor.                                  |
| `allComponentsSelector`       | none                                     | Every component, every iteration.                                                             |
| `improvementAcceptance`       | `{ minImprovement = 0 }`                 | Accepts a child whose minibatch total beats its parent's.                                     |
| `pairedPermutationAcceptance` | `{ alpha = 0.2, maxExact = 16 }`         | Accepts only when the paired improvement survives a sign-flip test.                           |
| `fullEvaluationPolicy`        | none                                     | Scores every validation instance per accepted candidate.                                      |
| `subsampledEvaluationPolicy`  | `{ size }`                               | Scores `size` instances, trading frontier fidelity for rollouts.                              |
| `lowerBoundEvaluationPolicy`  | `{ z = 1 }`                              | Full coverage, but picks the winner by mean minus `z` standard errors.                        |

Each export is a factory. Selector and acceptance interfaces accept custom functions. A `ValEvaluationPolicy` is an object with `selectInstances` and `bestCandidate` methods.

`pairedPermutationAcceptance` and `lowerBoundEvaluationPolicy` exist for metrics whose readings vary between runs of the same text. Both are strictly more conservative than the defaults, and on a metric that does not vary that is pure cost: in the twenty-seed benchmark the pair drops GEPA from 0.729 to 0.175 on the noiseless task and ties it on the noisy one. A sign-flip test also needs a wide enough minibatch to say anything — over three instances the smallest p-value it can produce is 0.125, so at the default `minibatchSize` no proposal clears an `alpha` below that.

### Reflection prompts

| Export                        | What it asks for                                              |
| ----------------------------- | ------------------------------------------------------------- |
| `buildReflectionPrompt`       | The default: read the failures, write better text.            |
| `buildSimplifyPrompt`         | A shorter version that retains behavior needed by the batch.  |
| `buildGeneralizePrompt`       | Text that avoids details specific to the current failures.    |
| `buildRewritePrompt`          | A replacement written without preserving the current wording. |
| `diverseReflectionStrategies` | All four builders for use with `reflection.strategies`.       |

Each implements `ReflectionPromptBuilder`, defined as `(args: ReflectionPromptArgs) => string`.

### `createDemoProposer`

```ts
createDemoProposer({ components, minScore, maxDemos, render, fallback });
```

This `proposeNewTexts` implementation fills selected components with few-shot examples from the reflective dataset. It uses existing record inputs, outputs, and scores, so it requires no additional rollouts or reflection calls. Examples are deduplicated by input and appended to the parent's block. `fallback` handles components not listed in `components`.

### `createPipelineAdapter`

```ts
createPipelineAdapter({ modules, input, score, concurrency });
```

A `GepaAdapter` for a system built from several modules in sequence, where each module's instruction is its own candidate component. Each `PipelineModule` has a `component` and a `run({ instruction, input, datum, signal })`; the first module receives `input(datum)`, and each subsequent module receives the previous module's output. `score({ datum, output, steps })` returns a `ScoreResult` for the whole rollout.

The trajectory is a `PipelineTrace` of `PipelineStep` entries, and the reflective dataset gives each component only its own step. That attribution is the point: reflection is only as good as the evidence it sees, and a module needs what it received and produced, not the pipeline's input and final answer.

Feedback is end-to-end and every module receives the same string. A metric scores the final output, so nothing in a score alone says which module lost the point; `score` is handed the whole trace for callers who can attribute better. Errors from a module are not caught — a helper cannot tell a rate limit from a bug, so classify inside `run` and return a transient `ScoreResult`, or let `raiseOnError` decide.

### Results, events, and resuming

**`GepaResult`** extends `OptimizerResult` with `bestCandidateId`, the complete candidate pool, lineage and per-instance scores, `paretoFrontier`, `scoreMatrix`, `perObjectiveBest`, `iterations`, `reflectionCalls`, `cacheHits`, and the final `snapshot`.

**`stopReason`** is one of `"budgetExhausted"`, `"costExhausted"`, `"deadlineReached"`, `"reflectionBudgetExhausted"`, `"aborted"`, or `"maxIterations"`.

**`onEvent`** receives a discriminated `GepaEvent`: `start`, `iterationStart`, `evaluation`, `proposal`, `candidateAccepted`, `candidateRejected` (with `reason: "worse" | "notSelected"`), `error`, and `finish`.

**`onCheckpoint`** runs after seed evaluation and each iteration with a JSON-serializable `GepaSnapshot`. Pass it as `resumeFrom` to continue. A fingerprint prevents resuming with a different seed candidate, instance set, or random seed. Every optimizer here has the same three: `onCheckpoint`, `resumeFrom`, and a `snapshot` on the result. A snapshot handed back as `resumeFrom` is copied, never mutated by the run that continues from it.

## `textopt/simba`

```ts
import { SimbaOptimizer, buildAdvicePrompt, parseAdvice } from "textopt/simba";
```

SIMBA uses the base `Adapter`: it reads outputs, scores, and feedback and builds its own evidence, so it needs no `makeReflectiveDataset`. `SimbaTask` adds `reflect`, `demoComponents`, `instructionComponents`, `renderDemo`, `buildAdvicePrompt`, `sampler`, `instanceId`, `cache`, `onEvent`, `onCheckpoint`, and `resumeFrom`.

| Option                 | Default              | Effect                                                                     |
| ---------------------- | -------------------- | -------------------------------------------------------------------------- |
| `minibatchSize`        | `32`                 | Instances per step. Must not exceed the trainingSet.                       |
| `candidates`           | `6`                  | Programs sampled per step, and the ceiling on candidates built from them.  |
| `maxSteps`             | `8`                  | Steps to run.                                                              |
| `maxDemos`             | `4`                  | Demos a candidate may hold before the loop starts dropping them.           |
| `samplingTemperature`  | `0.2`                | Sharpness of the pick between programs when sampling trajectories.         |
| `candidateTemperature` | `0.2`                | Sharpness of the pick between programs when choosing what to mutate.       |
| `strategies`           | both, or rules alone | Pins the mutation to `"appendDemo"`, `"appendRule"`, or leaves both drawn. |
| `maxReflectionCalls`   | unbounded            | Advice calls the run may make. Bounded separately from rollouts.           |
| `seed`                 | `0`                  | Seeds the run's random stream.                                             |
| `checkpointCache`      | `true`               | Include cached scores in every snapshot.                                   |
| `trackBestOutputs`     | `false`              | Keep the winner's validation outputs.                                      |

**What a run costs.** `(candidates + 1) × minibatchSize` per step, on top of `min(candidates + 1, maxSteps + 1) × validationSet.length` reserved before the first step for the finalist sweeps. The seed is scored as one of those finalists rather than up front, so every rollout a SIMBA run makes is one of these two.

Each step samples `candidates` programs from the pool over one minibatch, groups the results by instance, and ranks the instances by how much the programs disagreed — max-to-min gap first, then best score, then max-to-avg gap. Wide disagreement is a controlled experiment with the input held fixed, so the difference in reward is attributable to behaviour rather than to difficulty.

Two mutations are drawn at random per instance. `appendDemo` keeps the winning rollout as a few-shot example and costs no model call; it requires `demoComponents` and is unavailable without one. `appendRule` shows the better and worse run to `reflect` and appends the returned advice to each instruction component. Neither replaces text, so demos are dropped at a Poisson rate to stop a growing block from crowding out the instruction. When the advice budget is spent, `appendDemo` carries the run alone, or the run stops with `"reflectionBudgetExhausted"` if it was the only mutation enabled.

Guards keep uninformative contrasts out: a winner below the batch's tenth percentile is not a success to imitate, and a loser above the ninetieth is not a failure to avoid. When the two runs tied, the uninformative side is withheld and the model advises from one trajectory.

Only the step winners are scored on the full validation set, sampled evenly across the run so early winners stay in the running — minibatch scores are noisy and the genuine best is often not the most recent. Those rollouts are reserved before the search starts, so a small `maxMetricCalls` buys fewer steps than the arithmetic suggests.

**`buildAdvicePrompt`** implements `AdvicePromptBuilder` and asks for one `<advice component="name">…</advice>` block per component. **`parseAdvice(response)`** reads them back, ignoring prose written around them; a component the model had nothing to say about is absent rather than empty.

`SimbaResult` adds `seedScore`, `steps`, `finalists` (the step winners with their validation scores, best first), `reflectionCalls`, and `cacheHits`. `stopReason` is `"budgetExhausted"`, `"costExhausted"`, `"deadlineReached"`, `"reflectionBudgetExhausted"`, `"maxSteps"`, or `"aborted"`.

`textopt/simba` also exports the pure functions the loop is built from: `buildBuckets`, `percentile`, `softmaxWeights`, `topKPlusBaseline`, `samplePoisson`, and `evenlySpacedIndices`.

Ported from DSPy's SIMBA with two deliberate changes. A trajectory sample runs one program across the whole minibatch rather than resampling a program per instance: the adapter owns decoding here, so there is no temperature knob to vary and the variability comes from the program pool. And the percentile guards are strict rather than inclusive, because on a step where every rollout ties an inclusive guard blocks every mutation and the run does nothing at all — the one case where the guard's own premise does not hold.

## `textopt/bootstrap-search`

```ts
import { BootstrapSearchOptimizer } from "textopt/bootstrap-search";
```

DSPy's `BootstrapFewShotWithRandomSearch`. It uses the base `Adapter` and no reflection model at all: every candidate is assembled from outputs the system itself produced, so the search costs rollouts and nothing else. `BootstrapSearchTask` adds `demoComponents` (required), `renderDemo`, `goldOutput`, `instanceId`, `cache`, `onEvent`, `onCheckpoint`, and `resumeFrom`.

| Option             | Default | Effect                                                               |
| ------------------ | ------- | -------------------------------------------------------------------- |
| `candidates`       | `16`    | Shuffled harvests attempted, beyond the fixed candidates.            |
| `maxDemos`         | `4`     | Most demos a harvested set may hold.                                 |
| `minDemos`         | `1`     | Fewest demos a shuffled harvest may ask for.                         |
| `maxLabeledDemos`  | `16`    | Most demos the labels-only candidate may hold.                       |
| `demoMinScore`     | unset   | Score a rollout must reach to be kept. Unset keeps any rewarded one. |
| `stopAtScore`      | unset   | Stop as soon as a candidate reaches this validation score.           |
| `seed`             | `0`     | Seeds the run's random stream.                                       |
| `checkpointCache`  | `true`  | Include cached scores in every snapshot.                             |
| `trackBestOutputs` | `false` | Keep the winner's validation outputs.                                |

**What a run costs.** One validation sweep per candidate — `candidates` shuffled harvests after the two or three fixed ones — plus the training rollouts each harvest spends looking for demos, bounded by the size of `trainingSet`. A stricter `demoMinScore` raises that second number rather than lowering it, because a harvest keeps rolling out instances until it holds `maxDemos` of them or has run out of set.

Candidates are tried cheapest and most reliable first, following DSPy's special seeds: zero-shot, then labels-only when `goldOutput` is supplied, then one unshuffled full-size harvest, then the shuffled ones. A run cut short by its budget therefore still has the baseline it needs to report against.

Zero-shot stays in the running throughout. Demonstrations can hurt, and a search that cannot return "no demos" has no baseline. The labels-only candidate costs no rollout to build and is the only candidate available at all to a system too weak to bootstrap from.

Sizes vary across the shuffled harvests because more demos is not monotonically better: a long block crowds out the instruction, and which length wins is what this search settles. A harvest and its validation sweep are treated as one purchase, so the run never harvests demos it cannot afford to score.

Unlike DSPy, which bootstraps each predictor separately from the traces of one pass, this adapter interface runs the whole system: a harvest is a set of end-to-end rollouts and every demo component receives the same block. For per-module demos, use `createPipelineAdapter` with GEPA.

`BootstrapSearchResult` adds `seedScore`, `candidates` (each with its `source`, demo count, and score), `bootstrapMetricCalls`, and `cacheHits`. `stopReason` is `"budgetExhausted"`, `"costExhausted"`, `"deadlineReached"`, `"scoreReached"`, `"candidatesExhausted"`, or `"aborted"`.

## `textopt/opro`

```ts
import { OproOptimizer, buildOproPrompt } from "textopt/opro";
```

OPRO uses the base `Adapter`. `OproTask` adds `reflect` and optional `renderDatum`, `instanceId`, `cache`, and `onEvent` fields.

| Option               | Default           | Effect                                                     |
| -------------------- | ----------------- | ---------------------------------------------------------- |
| `proposalsPerRound`  | `8`               | Instructions drawn per round.                              |
| `concurrency`        | `1`               | How many may be in flight at once.                         |
| `maxRounds`          | `Infinity`        | Round ceiling.                                             |
| `maxReflectionCalls` | unbounded         | Bounded separately; no metric budget covers reflection.    |
| `historySize`        | `20`              | Maximum scored attempts included in the prompt.            |
| `exemplars`          | `3`               | Task inputs shown for grounding, redrawn each round.       |
| `scoringSetSize`     | unset             | Instances drawn once from the trainingSet to screen on.    |
| `fullEvalInterval`   | `3`               | Rounds between full validationSet sweeps of the incumbent. |
| `scoreScale`         | `100`             | What scores are multiplied by before being shown.          |
| `seed`               | `0`               | Seeds the run's random stream.                             |
| `buildPrompt`        | `buildOproPrompt` | Replaces the meta-prompt template.                         |

**What a run costs.** One validation sweep for the seed, then up to `proposalsPerRound` screenings per round, each over `validationSet` or over `scoringSetSize` instances when that is set, plus a full sweep of the incumbent every `fullEvalInterval` rounds. Rounds are also capped at `maxReflectionCalls / proposalsPerRound`, and a round advances one component in turn, so a candidate with several components needs proportionally more of them.

By default, every proposal is evaluated against the full `validationSet`; eight proposals on 500 instances require 4,000 rollouts. When `scoringSetSize` is set, proposals are screened on one fixed subset of `trainingSet`. The subset is not resampled because OPRO compares scores across proposals. Every `fullEvalInterval` rounds, the incumbent is evaluated on the full validation set.

Measured on 30 training and 30 validation instances over 8 rounds of 4 proposals, averaged across 10 seeds:

| `scoringSetSize`             | Best score | Rollouts |
| ---------------------------- | ---------- | -------- |
| unset (whole validation set) | 1.000      | 738      |
| 12                           | 1.000      | 383      |
| 6                            | 0.950      | 217      |
| 3                            | 0.862      | 137      |

In this sweep, 12 screening instances cut rollout count by 48% without changing the best score. Three instances reduced cost further but also reduced mean quality. Full-set scoring remains the default.

The reference implementation selects its winner by training-subset score. textopt instead returns the best candidate evaluated on the full validation set, so `bestScore` never falls below `seedScore`. Screening scores still guide the search but cannot determine the returned winner.

`OproResult` adds `seedScore`, `rounds`, `trajectory` (every candidate scored, in order), `reflectionCalls`, and `cacheHits`. `stopReason` is `"budgetExhausted"`, `"costExhausted"`, `"deadlineReached"`, `"reflectionBudgetExhausted"`, `"proposalsExhausted"`, `"maxRounds"`, or `"aborted"`.

`"proposalsExhausted"` means several rounds in a row produced only texts already tried. Such a round spends no rollouts, so without this the budget would never run down and a proposal model that has settled on one answer would loop forever.

History is sorted by ascending score so the strongest attempt appears nearest the request. Scores are shown as integers because models distinguish 41 from 68 more reliably than 0.41 from 0.68.

For multi-component candidates, each attempt records the other components present when it was scored. The prompt only includes attempts from the current component context, avoiding comparisons between scores obtained with different companion text. Single-component candidates require no filtering.

## `textopt/mipro`

```ts
import { MiproOptimizer, proposeConfiguration } from "textopt/mipro";
```

MIPRO uses the base `Adapter`. `MiproTask` adds `reflect`, `componentOptions`, `renderDatum`, `batchSampler`, `instanceId`, `cache`, and `onEvent`.

| Option                     | Default            | Effect                                                                             |
| -------------------------- | ------------------ | ---------------------------------------------------------------------------------- |
| `instructionsPerComponent` | `3`                | Menu entries generated per component, beyond the seed.                             |
| `minibatchSize`            | `35`               | Instances a trial is scored on. MIPROv2's `minibatch_size`.                        |
| `fullEvalInterval`         | `5`                | Trials between full evaluations. MIPROv2's `minibatch_full_eval_steps`.            |
| `demoSets`                 | `3`                | Bootstrapped demo sets generated per demo component.                               |
| `maxDemos`                 | `4`                | Demos in the largest generated set.                                                |
| `demoMinScore`             | unset              | Score a rollout must reach to be kept as a demo. Unset keeps any rewarded rollout. |
| `maxTrials`                | `30`               | Configurations evaluated.                                                          |
| `startupTrials`            | `10`               | Trials drawn uniformly before the surrogate takes over.                            |
| `gamma`                    | Optuna's rule      | Observations assigned to the good density: `ceil(10%)`, capped at 25.              |
| `surrogateSamples`         | `24`               | Candidate configurations sampled by the surrogate per trial.                       |
| `multivariate`             | `true`             | Model components jointly rather than one at a time.                                |
| `exemplars`                | `3`                | Task inputs shown when generating instructions.                                    |
| `datasetSummary`           | `true`             | Generate a `trainingSet` summary for the proposer. Uses one reflection call.       |
| `summaryExamples`          | `10`               | Training examples used to generate the summary.                                    |
| `tips`                     | built-in           | Style hints used when generating menu options.                                     |
| `buildPrompt`              | `buildMiproPrompt` | Replaces the proposal template.                                                    |

**What a run costs.** `maxTrials × minibatchSize` for the trials, plus `validationSet.length` for the seed and for each full evaluation — one every `fullEvalInterval` trials, and one at the end — plus up to `demoSets × trainingSet.length` to bootstrap the demo menus. The search stops as soon as a sweep is no longer affordable, because a trial that can never be promoted buys nothing. Trials should also scale with the menus they draw from: the space is their product, and the surrogate spends `startupTrials` of them sampling uniformly before it models anything.

The proposer receives the current text of other components, a generated summary of `trainingSet`, and task exemplars. The summary uses one reflection call and is skipped when all menus are supplied through `componentOptions`. Unlike DSPy's MIPROv2 implementation, textopt does not include program source because the adapter interface has no generic representation for it.

`componentOptions` adds menu entries without reflection calls. Each component's menu starts with its seed text, followed by these options.

`demoComponents` identifies components that contain few-shot blocks. Their menus are bootstrapped from successful `trainingSet` rollouts instead of generated by `reflect`. Each demo set uses a separate shuffled pass and a size between one and `maxDemos`; the zero-shot option is always included. Separate passes can produce different demos for stochastic systems but add redundant work for deterministic systems, matching MIPROv2's behavior. Bootstrap rollouts count against `maxMetricCalls` and are also reported as `bootstrapMetricCalls`.

When `goldOutput` is supplied, each demo component also includes a labels-only `LabeledFewShot` option. It requires no rollout because the expected output is provided by the caller, and remains available when the seed system cannot produce successful bootstrap examples.

Every `fullEvalInterval` trials, the unswept configuration with the highest average minibatch score receives a full evaluation. A final sweep runs when the search ends. Averaging repeated minibatch scores reduces sensitivity to individual samples.

The surrogate includes both minibatch and full-evaluation observations, matching DSPy. The seed's full score is registered before the first trial. In a ten-seed sweep, adding promoted configurations' full scores reduced repeat proposals of disproved configurations from 21 to 10. Repeated high minibatch scores can still outweigh one low validation score.

One difference from MIPROv2 is that textopt keeps labelled demo sets as separate menu options; MIPROv2 pads bootstrapped sets with labelled examples.

`MiproResult` adds `seedScore`, `trials`, `menu`, `observations`, `fullEvaluations`, `bootstrapMetricCalls`, `reflectionCalls`, and `cacheHits`. Only full validation evaluations update the incumbent; minibatch scores select configurations for full evaluation. `stopReason` is `"budgetExhausted"`, `"costExhausted"`, `"deadlineReached"`, `"maxTrials"`, or `"aborted"`.

A `MiproSnapshot` carries the option menus alongside the usual budget and RNG state. The menus matter most: building them is the expensive half of a run — a reflection call per instruction and a harvesting pass per demo set — and they are also what every trial's choice vector indexes into, so a resumed run that rebuilt them would both pay twice and reinterpret every observation it had already made.

**`proposeConfiguration({ observations, menuSizes, gamma, samples, startupTrials, multivariate, rng })`** exposes the TPE surrogate separately. It splits observations into good and remaining groups, models each density, samples from the good density, and ranks samples by the log density ratio.

With `multivariate: true`, each density is a mixture of kernels centered on observed configurations. This preserves dependencies between component options. With `multivariate: false`, each component uses an independent smoothed histogram, which learns from fewer trials but cannot represent interactions.

Measured on an objective where each component pays off only when the preceding one is also correct, across 15 seeds: in a space of 3,125 configurations with 60 trials, the joint model reached a mean best score of 0.91 and solved 10 runs, against 0.80 and 6 for the independent model. In a 16-configuration space with 30 trials over 20 seeds, the joint model found the optimum in every run, against 16 for the independent model. The multivariate model is the default because it leads on both.

Before `startupTrials`, or when all observations have the same score, sampling is uniform. TPE's good/bad split requires ranked observations; using it on tied scores can repeatedly select configurations already evaluated. This tie handling differs from published TPE and Optuna.

## `textopt/random-search`

```ts
import { RandomSearchOptimizer } from "textopt/random-search";
```

| Option        | Default                 | Effect                                            |
| ------------- | ----------------------- | ------------------------------------------------- |
| `variants`    | `4`                     | Variants drawn per round, each evaluated in full. |
| `concurrency` | `1`                     | How many may be in flight at once.                |
| `maxRounds`   | `Infinity`              | Round ceiling.                                    |
| `buildPrompt` | `buildParaphrasePrompt` | Replaces the paraphrase template.                 |

**What a run costs.** One validation sweep for the seed, then `variants × validationSet.length` per round. A round starts only when all of it is affordable, so a half-funded round never spends rollouts on variants the rest cannot be compared against.

`RandomSearchResult` adds `seedScore`, `rounds`, `variantsEvaluated`, `reflectionCalls`, and `cacheHits`. `stopReason` is `"budgetExhausted"`, `"costExhausted"`, `"deadlineReached"`, `"maxRounds"`, `"proposerStalled"`, or `"aborted"`.

`"proposerStalled"` is the guard against a livelock: a proposer that returns duplicates, or texts already in the cache, spends no budget at all, so with `maxRounds` unset the loop would spin forever burning reflection calls. A full pass over the components that buys neither a rollout nor an improvement ends the run.

The paraphrase prompt receives no score or feedback. Compare random search with a reflective optimizer under the same metric budget to measure the effect of reflection.

## `textopt/file-cache`

```ts
import { createFileCache } from "textopt/file-cache";

const cache = createFileCache({ path: ".textopt/scores.jsonl" });
```

An `EvaluationCache` that outlives the process, as an append-only JSONL log. A long run against a real provider is measured in hours and dollars, and an in-memory cache throws all of it away when the run ends.

Append-only rather than rewritten: a score is never invalidated, because the key names the candidate, the instance, and the environment — and a log survives a process killed mid-write, which a file rewritten in place does not. A record that does not parse is dropped rather than fatal, since the last line of an interrupted log is routinely half-written. Later records win, so a re-measured instance replaces its earlier reading.

`maxEntries` (default 1,000,000) bounds what is held in memory; the file itself is never trimmed. `entries()` is deliberately absent — it exists so a checkpoint can carry scores that would otherwise be lost, and these are already on disk.

This is the only entry point that imports `node:fs`.

## `textopt/testing`

Deterministic keyword-coverage fixtures for testing optimizers and adapters without network access.

| Export                          | Description                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| `KEYWORD_EXAMPLES`              | Four `KeywordExample` rows.                                                         |
| `createKeywordAdapter()`        | A `GepaAdapter` scoring how many required terms the candidate text covers.          |
| `createKeywordReflector()`      | A `TextModel` that folds the missing terms into the current instruction.            |
| `createSamplingReflector()`     | A `TextModel` that ignores the prompt and appends a term from a fixed pool.         |
| `createHillClimbingReflector()` | A `TextModel` that reads score history and extends the best attempt.                |
| `SAMPLING_POOL`                 | Useful terms interleaved with distractors, the pool the two above draw from.        |
| `createDegradingReflector()`    | A `TextModel` that always proposes something strictly worse, for testing rejection. |
| `buildReflectionPrompt`         | The default prompt template, also exported from `textopt/gepa`.                     |

## Adapters

Prebuilt adapters for common stacks live alongside this package: [`@textopt/ai-sdk`](https://github.com/ctdio/textopt/tree/main/packages/ai-sdk), [`@textopt/langchain`](https://github.com/ctdio/textopt/tree/main/packages/langchain), and [`@textopt/braintrust`](https://github.com/ctdio/textopt/tree/main/packages/braintrust).

## License

MIT
