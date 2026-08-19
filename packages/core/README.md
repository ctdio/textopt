# textopt

Framework-agnostic prompt optimization for TypeScript: GEPA, OPRO, MIPRO, and a random-search baseline behind one `Optimizer` contract.

Zero dependencies. This is the API reference for the package. For what each algorithm is, how the searches work, and which to reach for, see the [project README](../../README.md).

## Entry points

| Import                  | Contains                                                                                                                                                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `textopt`               | The optimizer and adapter contracts, the shared evaluator, demo bootstrapping, the evaluation cache, and a concurrency helper. No engine: importing `GepaOptimizer` from here will fail. |
| `textopt/gepa`          | `GepaOptimizer` and everything GEPA-specific. Its config, task, result, adapter, events, snapshot, and the swappable strategies.                                                         |
| `textopt/opro`          | `OproOptimizer` and its config, task, result, and events. Scalar scores only; no reflective dataset.                                                                                     |
| `textopt/mipro`         | `MiproOptimizer`, its types, and `proposeConfiguration` — the TPE surrogate, usable on its own.                                                                                          |
| `textopt/random-search` | `RandomSearchOptimizer` and its types. The uninformed control the others are measured against.                                                                                           |
| `textopt/testing`       | A deterministic, LLM-free system under optimization and reflection model, for exercising the loop or your own adapter without a network.                                                 |

## `textopt`

```ts
import {
  bootstrapDemos,
  componentNames,
  createEvaluator,
  createMemoryCache,
  formatDemos,
  mapWithConcurrency,
  parseDemos,
  parseProposedText,
} from "textopt";
import type {
  Adapter,
  BatchSampler,
  BootstrapResult,
  Candidate,
  CachedScore,
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
  Optimizer,
  OptimizerResult,
  OptimizerTask,
  Rng,
  ScoreResult,
  TextModel,
} from "textopt";
```

### Contracts

**`Candidate<K extends string>`** is `Record<K, string>`: the unit of optimization, a map of named text components. `K` is inferred from the seed candidate, so a misspelled component is a compile error.

**`Adapter<Datum, Traj, Out, K>`** is the single integration seam. One method:

```ts
evaluate(args: EvaluateArgs<Datum, K>): Promise<EvaluationBatch<Traj, Out>> | EvaluationBatch<Traj, Out>
```

`EvaluateArgs` carries the `batch`, the `candidate`, a `captureTraces` flag, an `EvaluationContext` under `run` (`iteration`, `phase`, `split`, `candidateId`) to forward into your own tracing, and an optional `signal`.

`EvaluationBatch` returns `outputs` and `scores`, one per instance, higher is better. Optional alongside them: `feedback`, `trajectories`, `objectiveScores`, `transient`. Feedback is what a reflective optimizer reads, and without it the search is blind.

**`ScoreResult`** is what a per-instance scorer returns: `score`, plus optional `feedback`, `objectiveScores`, and `transient`. It is shared across adapters, so a scorer written for one works in another.

**`transient`** marks a score caused by infrastructure rather than by the candidate: a rate limit, a 5xx, a network blip. Transient scores are never cached, so an outage cannot pin a candidate to a permanent zero.

**`Optimizer<Stop extends string>`** has exactly one method, `optimize(task: OptimizerTask) => Promise<OptimizerResult>`. `OptimizerTask` is the run-level input every optimizer needs (`seedCandidate`, `trainset`, `valset`, `testset`, `adapter`, `maxMetricCalls`, `signal`). `OptimizerResult` is what every optimizer reports (`bestCandidate`, `bestScore`, `bestOutputs`, `metricCalls`, `testScore`, `testMetricCalls`, `stopReason`). An optimizer's own task and result types extend these.

**`testset`** is held out of the search entirely and scored once, on the winner, after the run ends. Selection pressure is applied to the valset throughout, so `bestScore` is partly fitted to it; `testScore` is the only number in a result that no candidate was ever selected against. Those rollouts are measurement rather than search, so they are reported as `testMetricCalls` and are not charged against `maxMetricCalls`.

**`TextModel`** is `({ prompt, signal }) => Promise<string>`, the whole provider seam.

### Values

**`createMemoryCache({ maxEntries = 100_000, entries })`** returns an in-memory `EvaluationCache`. `entries` restores a previous run's checkpointed scores. Eviction is oldest-first once `maxEntries` is reached.

To back the cache with Redis, SQLite, or a file, implement **`EvaluationCache`** yourself: `get(key)`, `set(key, cached)`, and optionally `entries()`. Leave `entries()` off a store that is already durable and has nothing to checkpoint.

**`mapWithConcurrency({ items, limit, task, signal })`** is an order-preserving map with a concurrency limit. Adapters use it to fan a batch across a bounded number of in-flight model calls. Work already running settles before a failure propagates, so a rejected run does not keep spending budget in the background.

**`componentNames(candidate)`** is `Object.keys` that preserves the component union instead of widening it back to `string`.

**`createEvaluator({ adapter, budget, cache, trackOutputs, onEvaluation, signal })`** is the engine every optimizer in this package runs on: it dispatches to the adapter, prices batches against the cache before spending, charges the budget, drops transient scores, and emits an `EvaluationEvent` per batch. `evaluate` returns a `ScoredBatch`; `evaluateTraced` returns the traced `EvaluationBatch`, or `null` when the budget cannot cover it. `BudgetExhausted` is thrown when a charged batch outruns the remaining calls. Writing an optimizer against this is what keeps budget accounting and caching identical across all four.

**`bootstrapDemos({ adapter, candidate, trainset, minScore, maxDemos, batchSize, maxMetricCalls, rng, renderDemo, signal })`** runs a candidate over the trainset and keeps the rollouts that scored at or above `minScore`, returning the `demos`, a formatted `block` ready to drop into a candidate, and what it spent. It always evaluates fresh, because the cache stores scores and this needs outputs.

**`formatDemos(demos, { render })`** and **`parseDemos(text)`** are the two halves of the block format (`<demo>`, `<input>`, `<output>`), so a demo component survives a round trip through a candidate and back.

**`parseProposedText(text)`** pulls the proposal out of a reflection response, tolerating fenced blocks and surrounding commentary. Shared by every optimizer that asks a model for text.

**`BatchSampler<Datum>`** and **`Rng`** are exported as types only. Both default implementations are internal, so replacing one means implementing the interface.

## `textopt/gepa`

```ts
import { GepaOptimizer } from "textopt/gepa";

const gepa = new GepaOptimizer(config); // GepaConfig: stateless, reusable
const result = await gepa.optimize(task); // GepaTask: one problem
```

An optimizer holds no run state, so one instance can run any number of problems.

### `GepaConfig`

How the search runs. Free of your types, and safe to share across runs.

| Option                     | Default                   | Effect                                                                                                                                  |
| -------------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `minibatchSize`            | `3`                       | Instances per screening batch.                                                                                                          |
| `maxIterations`            | `Infinity`                | Iteration ceiling. The metric budget usually binds first.                                                                               |
| `seed`                     | `0`                       | Seeds the run's random stream.                                                                                                          |
| `candidateSelector`        | `paretoSelector()`        | Which candidate becomes the next parent.                                                                                                |
| `acceptance`               | `improvementAcceptance()` | Whether a screened child beats its parent.                                                                                              |
| `merge.enabled`            | components > 1            | System-aware merge of two lineages.                                                                                                     |
| `merge.maxInvocations`     | `5`                       | Ceiling on merges attempted per run.                                                                                                    |
| `merge.valOverlapFloor`    | `5`                       | Validation instances two lineages must share to be eligible. A valset smaller than this never merges.                                   |
| `skipPerfectScore`         | `true`                    | Skip reflection when the parent already scores perfectly on the minibatch.                                                              |
| `perfectScore`             | `1`                       | The per-instance score treated as leaving no room.                                                                                      |
| `rejectedProposalMemory`   | `3`                       | Rejected texts per component shown back to reflection. `0` disables it. An extension: not in the paper, and the only one on by default. |
| `proposals.perIteration`   | `1`                       | Mutations drawn per iteration, each with its own parent and minibatch.                                                                  |
| `proposals.concurrency`    | `1`                       | How many may be in flight at once.                                                                                                      |
| `proposals.selection`      | `"all"`                   | Which improving proposals to keep: `"all"`, `"best"`, or `{ keep: n }`.                                                                 |
| `reflection.maxCalls`      | unbounded                 | Hard ceiling on reflection calls. The run stops when it is reached.                                                                     |
| `reflection.maxRecords`    | unbounded                 | Records shown per component. The worst-scoring ones are kept.                                                                           |
| `reflection.maxCharacters` | unbounded                 | Rough ceiling on the serialized records.                                                                                                |
| `reflection.buildPrompt`   | `buildReflectionPrompt`   | Replaces the prompt template. Custom proposers ignore it.                                                                               |
| `reflection.strategies`    | none                      | Rotates over several prompt templates, one per proposal slot. Mutually exclusive with `buildPrompt`.                                    |
| `checkpointCache`          | `true`                    | Include cached scores in every snapshot.                                                                                                |
| `trackBestOutputs`         | `false`                   | Keep validation outputs so the winner's can be read back.                                                                               |
| `raiseOnError`             | `true`                    | Rethrow adapter failures instead of skipping the iteration.                                                                             |

### `GepaTask`

One problem, its data, and its IO. Extends `OptimizerTask`.

Required: `seedCandidate`, `trainset`, `adapter` (a `GepaAdapter`), `reflect` (a `TextModel`), and `maxMetricCalls`.

| Option                | Default                                                                              |
| --------------------- | ------------------------------------------------------------------------------------ |
| `valset`              | the trainset                                                                         |
| `testset`             | none. Held out of the search and scored once, on the winner                          |
| `componentSelector`   | `roundRobinComponentSelector()`                                                      |
| `batchSampler`        | an epoch-shuffled sampler over `minibatchSize`                                       |
| `valEvaluationPolicy` | `fullEvaluationPolicy()`                                                             |
| `instanceId`          | a content hash of the datum, falling back to its position when it will not serialize |
| `cache`               | a per-run memory cache. Pass `false` to disable                                      |

`onEvent`, `onCheckpoint`, `resumeFrom`, and `signal` have no defaults.

`seedCandidate` and `trainset` are the inference sites for the component names and the datum type. Every other position is `NoInfer`, so it gets checked against them rather than widening them.

### `GepaAdapter`

Extends `Adapter` with what reflection needs:

```ts
makeReflectiveDataset(args: MakeReflectiveDatasetArgs<Datum, Traj, Out, K>): ReflectiveDataset<K>
proposeNewTexts?(args: ProposeArgs<K>): ComponentPatch<K>   // optional
```

Both may be sync or async. `ReflectiveDataset` is a partial map of component name to `ReflectiveRecord[]`, each record holding `inputs`, `generatedOutputs`, `feedback`, `score`, and a typed `evidence` slot for whatever else the reflection model should read. It is partial because an adapter only fills the components it was asked to update.

Implementing `proposeNewTexts` replaces the reflection call entirely, which is how a run goes fully offline. `reflect` is still required by the type, so pass a stub.

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
| `fullEvaluationPolicy`        | none                                     | Scores every validation instance per accepted candidate.                                      |
| `subsampledEvaluationPolicy`  | `{ size }`                               | Scores `size` instances, trading frontier fidelity for rollouts.                              |

Each is a factory. The first three return a `CandidateSelector`, `ComponentSelector`, or `AcceptancePolicy`, so those seams take a plain function of your own instead. A `ValEvaluationPolicy` is an object with two methods, `selectInstances` and `bestCandidate`.

### Reflection prompts

| Export                        | What it asks for                                                              |
| ----------------------------- | ----------------------------------------------------------------------------- |
| `buildReflectionPrompt`       | The default: read the failures, write better text.                            |
| `buildSimplifyPrompt`         | The shortest text that still handles the batch. Prompts accrete; this prunes. |
| `buildGeneralizePrompt`       | Text that does not overfit the specific failures it was shown.                |
| `buildRewritePrompt`          | A fresh attempt, for a lineage that has stopped moving.                       |
| `diverseReflectionStrategies` | All four as a rotation, for `reflection.strategies`.                          |

Each is a `ReflectionPromptBuilder` — `(args: ReflectionPromptArgs) => string` — so your own drops into the same list.

### `createDemoProposer`

```ts
createDemoProposer({ components, minScore, maxDemos, render, fallback });
```

A `proposeNewTexts` implementation that fills the named components with few-shot examples harvested from the reflective dataset the adapter already built. Because the records carry `inputs`, `generatedOutputs`, and `score`, this costs no extra rollouts and no reflection call. Examples are deduped by input and appended to the block the parent already holds, so they accumulate along the accepted lineage. `fallback` handles every other component.

### Results, events, and resuming

**`GepaResult`** extends `OptimizerResult` with `bestCandidateId`, the full `candidates` pool (each a `CandidateRecord` with lineage, per-instance scores, and provenance), the `paretoFrontier`, the `scoreMatrix`, `perObjectiveBest`, `iterations`, `reflectionCalls`, `cacheHits`, and the final `snapshot`.

**`stopReason`** is one of `"budgetExhausted"`, `"reflectionBudgetExhausted"`, `"aborted"`, or `"maxIterations"`.

**`onEvent`** receives a discriminated `GepaEvent`: `start`, `iterationStart`, `evaluation`, `proposal`, `candidateAccepted`, `candidateRejected` (with `reason: "worse" | "notSelected"`), `error`, and `finish`.

**`onCheckpoint`** fires after the seed is scored and after every iteration, with a plain-JSON `GepaSnapshot`. Hand it back as `resumeFrom` to continue. Every snapshot is fingerprinted against its seed candidate, instance ids, and seed, so resuming against a different setup throws instead of silently scoring old candidates against new data.

## `textopt/opro`

```ts
import { OproOptimizer, buildOproPrompt } from "textopt/opro";
```

Takes the base `Adapter`, not `GepaAdapter`. `OproTask` adds `reflect`, and optionally `renderDatum`, `instanceId`, `cache`, and `onEvent`.

| Option               | Default           | Effect                                                  |
| -------------------- | ----------------- | ------------------------------------------------------- |
| `proposalsPerRound`  | `8`               | Instructions drawn per round.                           |
| `concurrency`        | `1`               | How many may be in flight at once.                      |
| `maxRounds`          | `Infinity`        | Round ceiling.                                          |
| `maxReflectionCalls` | unbounded         | Bounded separately; no metric budget covers reflection. |
| `historySize`        | `20`              | Scored attempts the prompt carries, strongest kept.     |
| `exemplars`          | `3`               | Task inputs shown for grounding.                        |
| `scoringSetSize`     | unset             | Instances drawn once from the trainset to screen on.    |
| `fullEvalInterval`   | `3`               | Rounds between full valset sweeps of the incumbent.     |
| `scoreScale`         | `100`             | What scores are multiplied by before being shown.       |
| `seed`               | `0`               | Seeds the run's random stream.                          |
| `buildPrompt`        | `buildOproPrompt` | Replaces the meta-prompt template.                      |

`scoringSetSize` is the paper's economics. By default every proposal is measured on the whole `valset`, which is the reliable reading and the expensive one — eight proposals against a 500-instance valset is 4000 rollouts before anything is learned. Set it and proposals are screened on a slice of the trainset drawn **once** and never resampled, since the meta-prompt asks the model to read a gradient across scores and a gradient across different instances is not one. Every `fullEvalInterval` rounds the incumbent is swept on the full valset.

Measured on 30 training and 30 validation instances over 8 rounds of 4 proposals, averaged across 10 seeds:

| `scoringSetSize`     | Best score | Rollouts |
| -------------------- | ---------- | -------- |
| unset (whole valset) | 1.000      | 738      |
| 12                   | 1.000      | 383      |
| 6                    | 0.950      | 217      |
| 3                    | 0.862      | 137      |

Screening on 12 of 30 instances costs half as much and gives up nothing; screening on 3 gives up a great deal. The knob is the trade, and it is unset by default because a run that screens on too little converges confidently on the wrong thing.

One deliberate difference in _reporting_, not search. The paper's search picks its winner by training-subset score; this reports the best candidate a full sweep has actually seen, so `bestScore` is always a number the valset produced and never drops below `seedScore`. A candidate that screens well and sweeps badly still steers the search — it just cannot be handed back as the answer.

`OproResult` adds `seedScore`, `rounds`, `trajectory` (every candidate scored, in order), `reflectionCalls`, and `cacheHits`. `stopReason` is `"budgetExhausted"`, `"reflectionBudgetExhausted"`, `"aborted"`, or `"maxRounds"`.

The history is rendered in **ascending** score order, so the strongest attempt sits closest to the request. Scores are scaled to integers because models discriminate between 41 and 68 far more reliably than between 0.41 and 0.68.

Generalizing OPRO past a single string needs one guard the paper does not: an attempt is recorded with the text of every _other_ component at the time it was scored, and only attempts measured in the current context are shown. Without that, a score earned before another component moved sits in the list beside current ones, and the gradient the model is asked to read spans measurements that were never comparable. With a one-component candidate — the paper's case — nothing is ever filtered.

## `textopt/mipro`

```ts
import { MiproOptimizer, proposeConfiguration } from "textopt/mipro";
```

Takes the base `Adapter`. `MiproTask` adds `reflect` and `componentOptions`, plus the usual `renderDatum`, `batchSampler`, `instanceId`, `cache`, and `onEvent`.

| Option                     | Default            | Effect                                                                                                                |
| -------------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `instructionsPerComponent` | `3`                | Menu entries generated per component, beyond the seed.                                                                |
| `minibatchSize`            | `5`                | Instances a trial is scored on.                                                                                       |
| `fullEvalInterval`         | `6`                | Trials between full sweeps of the best-average config. dspy spells the same cadence as `minibatch_full_eval_steps=5`. |
| `demoSets`                 | `3`                | Bootstrapped demo sets generated per demo component.                                                                  |
| `maxDemos`                 | `4`                | Demos in the largest generated set.                                                                                   |
| `demoMinScore`             | `1`                | Score a rollout must reach to be kept as a demo.                                                                      |
| `maxTrials`                | `30`               | Configurations evaluated.                                                                                             |
| `startupTrials`            | `10`               | Trials drawn uniformly before the surrogate takes over.                                                               |
| `gamma`                    | Optuna's rule      | Fraction treated as good. Unset means a tenth of them, capped at 25, which narrows as the run goes on.                |
| `surrogateSamples`         | `24`               | Configurations drawn per trial, best of batch proposed.                                                               |
| `multivariate`             | `true`             | Model components jointly rather than one at a time.                                                                   |
| `exemplars`                | `3`                | Task inputs shown when generating instructions.                                                                       |
| `datasetSummary`           | `true`             | Summarize the trainset once and show it to the proposer. One reflection call.                                         |
| `summaryExamples`          | `10`               | Trainset entries the summary is written from.                                                                         |
| `tips`                     | built-in           | Style hints, so a menu spreads over approaches.                                                                       |
| `buildPrompt`              | `buildMiproPrompt` | Replaces the proposal template.                                                                                       |

The proposer is grounded the way MIPROv2's is, rather than being handed a component's text in isolation. It sees the other components' current text, so an instruction is written to sit alongside its neighbours instead of duplicating or contradicting them; it sees a summary of the trainset written by one reflection call over more entries than the exemplars can fit, so it can aim at the task rather than at three instances of it; and it sees the exemplars themselves. The summary call is skipped when every menu was supplied and nothing needs proposing. Not covered: dspy also shows the proposer the program's source, which has no generic analogue here.

`componentOptions` supplies menu entries **verbatim**, with no reflection call. The menu for a component is always its seed text followed by its options.

`demoComponents` names the components holding few-shot blocks, and their menus are bootstrapped from the trainset instead of written by `reflect` — this is MIPROv2's second search dimension, instructions and demonstrations optimized together. Each set gets its own harvesting pass over a freshly shuffled trainset, at sizes spanning one demo up to `maxDemos`; the zero-shot option is always kept on the menu, since demos can hurt. Independent passes only matter when the system is stochastic — a second pass can turn a previously failing example into a demo — and cost rollouts for nothing when it is not, which is the trade MIPROv2 makes. Those rollouts are charged to `maxMetricCalls` and reported separately as `bootstrapMetricCalls`.

Supply `goldOutput` and every demo component also keeps a labels-only set on its menu, which is MIPROv2's `LabeledFewShot` candidate. It is the one demo set that costs nothing — the output is known rather than produced — and it is the only one a system too weak to bootstrap from can still offer. Only the caller can provide it; nothing generic can tell which part of a datum is the answer.

Promotion follows MIPROv2's cadence: every `fullEvalInterval` trials the configuration with the best **average** minibatch reading that has not been swept yet is evaluated in full, plus a final sweep when the run ends. Averaging repeated readings is the point — screening on a single minibatch lets one lucky draw decide the run.

The surrogate hears full sweeps as well as minibatch readings, as dspy's does. The seed's sweep is registered before the first trial, so the search starts from a measured reference point rather than buying one, and every promoted configuration's full score is added when it lands. That last part is a smaller correction than it sounds: across ten seeds it cut how often a swept-and-disproved configuration came back from 21 draws to 10, but a configuration that reads high on _every_ minibatch and low on the valset keeps being proposed regardless — one authoritative score does not outvote a dozen misleading ones.

One deliberate gap against MIPROv2: it pads bootstrapped sets with labelled examples to a combined size, where this keeps the labelled set separate on the menu so the search can tell the two apart.

`MiproResult` adds `seedScore`, `trials`, `menu` (the space actually searched), `observations`, `fullEvaluations`, `bootstrapMetricCalls`, `reflectionCalls`, and `cacheHits`. Only a full validation sweep can move the incumbent; minibatch readings decide what is worth sweeping.

**`proposeConfiguration({ observations, menuSizes, gamma, samples, startupTrials, multivariate, rng })`** is the surrogate on its own: it splits the observations into good and rest, models the density of each, samples from the good one and ranks the draws by the log ratio.

Under `multivariate` — the default, and what MIPROv2 turns on in Optuna — each density is a mixture with one kernel centred on each observed configuration. A kernel keeps a configuration's components together, so "this option works, but only beside that one" survives into the proposal. Turning it off models each component with its own smoothed histogram instead: evidence about a component then generalizes across every combination it appears in, which converges from far fewer trials but cannot express a dependency at all.

Which is better depends on how much of the space a run can cover. Measured on five components of five options — 3125 configurations against 60 trials — the joint model reaches a mean best of 0.87 and solves 8 runs in 15, where the independent one reaches 0.78 and solves 5. On a 16-configuration space that 30 trials nearly enumerate, the ordering reverses: independent lands on the best configuration in 16 runs of 20 against 14. Joint is the default because the case it loses is the case that needed a surrogate least.

Below `startupTrials`, or while every observation carries the same score, it samples uniformly instead — a good/bad split presumes the scores rank the observations, and acting on that presumption when they do not locks the search onto configurations it has already measured. Neither published TPE nor Optuna guards the tie; this one is a deliberate departure.

## `textopt/random-search`

```ts
import { RandomSearchOptimizer } from "textopt/random-search";
```

| Option        | Default                 | Effect                                            |
| ------------- | ----------------------- | ------------------------------------------------- |
| `variants`    | `4`                     | Variants drawn per round, each evaluated in full. |
| `concurrency` | `1`                     | How many may be in flight at once.                |
| `maxRounds`   | `Infinity`              | Round ceiling.                                    |
| `seed`        | `0`                     | Seeds the run's random stream.                    |
| `buildPrompt` | `buildParaphrasePrompt` | Replaces the paraphrase template.                 |

`RandomSearchResult` adds `seedScore`, `rounds`, `variantsEvaluated`, `reflectionCalls`, and `cacheHits`.

The paraphrase prompt states outright that it has no information about how the current text scored. That ablation is the point: run it on the same budget as a reflective optimizer and the difference is what reflection bought.

## `textopt/testing`

A keyword-coverage task with a deterministic gradient, plus reflection models to match. No network, milliseconds per run.

| Export                          | What it is                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `KEYWORD_EXAMPLES`              | Four `KeywordExample` rows.                                                                             |
| `createKeywordAdapter()`        | A `GepaAdapter` scoring how many required terms the candidate text covers.                              |
| `createKeywordReflector()`      | A `TextModel` that folds the missing terms into the current instruction.                                |
| `createSamplingReflector()`     | A `TextModel` that ignores the prompt and appends a pooled term, for testing blind proposal.            |
| `createHillClimbingReflector()` | A `TextModel` that reads a score history and extends the best attempt, for testing score-driven search. |
| `SAMPLING_POOL`                 | Useful terms interleaved with distractors, the pool the two above draw from.                            |
| `createDegradingReflector()`    | A `TextModel` that always proposes something strictly worse, for testing rejection.                     |
| `buildReflectionPrompt`         | The default prompt template, also exported from `textopt/gepa`.                                         |

## Adapters

Prebuilt adapters for common stacks live alongside this package: [`@textopt/ai-sdk`](../ai-sdk), [`@textopt/langchain`](../langchain), and [`@textopt/braintrust`](../braintrust).

## License

MIT
