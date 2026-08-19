# textopt

Framework-agnostic prompt optimization for TypeScript: GEPA, OPRO, MIPRO, and a random-search baseline behind one `Optimizer` contract.

This package has no runtime dependencies. For an overview of the algorithms and guidance on choosing one, see the [project README](../../README.md).

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

**`Candidate<K extends string>`** is a `Record<K, string>` containing the text components to optimize. `K` is inferred from the seed candidate, so misspelled component names fail type checking.

**`Adapter<Datum, Trajectory, Output, K>`** connects an optimizer to the evaluated system:

```ts
evaluate(args: EvaluateArgs<Datum, K>): Promise<EvaluationBatch<Trajectory, Output>> | EvaluationBatch<Trajectory, Output>
```

`EvaluateArgs` contains the `batch`, `candidate`, `captureTraces`, an optional abort `signal`, and a `run` context with `iteration`, `phase`, `split`, and `candidateId`. The run context can be forwarded to a tracing system.

`EvaluationBatch` contains one output and score per instance. Higher scores are better. It may also include `feedback`, `trajectories`, `objectiveScores`, and `transient` flags. Reflective optimizers use `feedback` to generate revisions.

**`ScoreResult`** is the common return type for per-instance scorers: `score`, with optional `feedback`, `objectiveScores`, and `transient`.

**`transient`** marks scores caused by infrastructure failures such as rate limits, 5xx responses, or network errors. Transient scores are not cached.

**`Optimizer<Stop extends string>`** defines `optimize(task: OptimizerTask) => Promise<OptimizerResult>`. `OptimizerTask` contains the shared run inputs: `seedCandidate`, `trainingSet`, `validationSet`, `testSet`, `adapter`, `maxMetricCalls`, and `signal`. `OptimizerResult` contains `bestCandidate`, `bestScore`, `bestOutputs`, `metricCalls`, `testScore`, `testMetricCalls`, and `stopReason`. Optimizer-specific task and result types extend these interfaces.

**`testSet`** is excluded from search and evaluated once against the winner. Because candidates are selected on `validationSet`, `bestScore` may be fitted to it. `testScore` measures held-out performance. Test rollouts are reported as `testMetricCalls` and do not count against `maxMetricCalls`.

**`TextModel`** is the provider-independent interface `({ prompt, signal }) => Promise<string>`.

### Values

**`createMemoryCache({ maxEntries = 100_000, entries })`** returns an in-memory `EvaluationCache`. Pass `entries` to restore checkpointed scores. When full, the cache evicts the oldest entry.

For Redis, SQLite, or file-backed caching, implement **`EvaluationCache`** with `get(key)`, `set(key, cached)`, and optional `entries()`. A durable store does not need `entries()`.

**`mapWithConcurrency({ items, limit, task, signal })`** maps items in order while limiting concurrent tasks. Running tasks settle before an error is propagated.

**`componentNames(candidate)`** returns `Object.keys(candidate)` while preserving the component-name union.

**`createEvaluator({ adapter, budget, cache, trackOutputs, onEvaluation, signal })`** handles adapter calls, caching, budget accounting, transient scores, and evaluation events. `evaluate` returns a `ScoredBatch`. `evaluateTraced` returns an `EvaluationBatch`, or `null` when the remaining budget cannot cover the batch. A batch that exceeds the charged budget throws `BudgetExhausted`. All included optimizers use this evaluator.

**`bootstrapDemos({ adapter, candidate, trainingSet, minScore, maxDemos, batchSize, maxMetricCalls, rng, renderDemo, signal })`** evaluates a candidate on `trainingSet` and keeps rollouts scoring at least `minScore`. It returns the selected `demos`, a formatted `block`, and the metric calls used. It does not use the score cache because it needs rollout outputs.

**`formatDemos(demos, { render })`** and **`parseDemos(text)`** write and read the `<demo>`, `<input>`, and `<output>` block format.

**`parseProposedText(text)`** extracts a proposal from a reflection response, including responses with fenced blocks or surrounding commentary.

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

| Option                     | Default                   | Effect                                                                                                                                  |
| -------------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `minibatchSize`            | `3`                       | Instances per screening batch.                                                                                                          |
| `maxIterations`            | `Infinity`                | Iteration ceiling. The metric budget usually binds first.                                                                               |
| `seed`                     | `0`                       | Seeds the run's random stream.                                                                                                          |
| `candidateSelector`        | `paretoSelector()`        | Which candidate becomes the next parent.                                                                                                |
| `acceptance`               | `improvementAcceptance()` | Whether a screened child beats its parent.                                                                                              |
| `merge.enabled`            | components > 1            | System-aware merge of two lineages.                                                                                                     |
| `merge.maxInvocations`     | `5`                       | Ceiling on merges attempted per run.                                                                                                    |
| `merge.valOverlapFloor`    | `5`                       | Validation instances two lineages must share to be eligible. A validationSet smaller than this never merges.                            |
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
| `fullEvaluationPolicy`        | none                                     | Scores every validation instance per accepted candidate.                                      |
| `subsampledEvaluationPolicy`  | `{ size }`                               | Scores `size` instances, trading frontier fidelity for rollouts.                              |

Each export is a factory. Selector and acceptance interfaces accept custom functions. A `ValEvaluationPolicy` is an object with `selectInstances` and `bestCandidate` methods.

### Reflection prompts

| Export                        | What it asks for                                                              |
| ----------------------------- | ----------------------------------------------------------------------------- |
| `buildReflectionPrompt`       | The default: read the failures, write better text.                            |
| `buildSimplifyPrompt`         | The shortest text that still handles the batch. Prompts accrete; this prunes. |
| `buildGeneralizePrompt`       | Text that does not overfit the specific failures it was shown.                |
| `buildRewritePrompt`          | A fresh attempt, for a lineage that has stopped moving.                       |
| `diverseReflectionStrategies` | All four as a rotation, for `reflection.strategies`.                          |

Each implements `ReflectionPromptBuilder`, defined as `(args: ReflectionPromptArgs) => string`.

### `createDemoProposer`

```ts
createDemoProposer({ components, minScore, maxDemos, render, fallback });
```

This `proposeNewTexts` implementation fills selected components with few-shot examples from the reflective dataset. It uses existing record inputs, outputs, and scores, so it requires no additional rollouts or reflection calls. Examples are deduplicated by input and appended to the parent's block. `fallback` handles components not listed in `components`.

### Results, events, and resuming

**`GepaResult`** extends `OptimizerResult` with `bestCandidateId`, the complete candidate pool, lineage and per-instance scores, `paretoFrontier`, `scoreMatrix`, `perObjectiveBest`, `iterations`, `reflectionCalls`, `cacheHits`, and the final `snapshot`.

**`stopReason`** is one of `"budgetExhausted"`, `"reflectionBudgetExhausted"`, `"aborted"`, or `"maxIterations"`.

**`onEvent`** receives a discriminated `GepaEvent`: `start`, `iterationStart`, `evaluation`, `proposal`, `candidateAccepted`, `candidateRejected` (with `reason: "worse" | "notSelected"`), `error`, and `finish`.

**`onCheckpoint`** runs after seed evaluation and each iteration with a JSON-serializable `GepaSnapshot`. Pass it as `resumeFrom` to continue. A fingerprint prevents resuming with a different seed candidate, instance set, or random seed.

## `textopt/opro`

```ts
import { OproOptimizer, buildOproPrompt } from "textopt/opro";
```

Takes the base `Adapter`, not `GepaAdapter`. `OproTask` adds `reflect`, and optionally `renderDatum`, `instanceId`, `cache`, and `onEvent`.

| Option               | Default           | Effect                                                     |
| -------------------- | ----------------- | ---------------------------------------------------------- |
| `proposalsPerRound`  | `8`               | Instructions drawn per round.                              |
| `concurrency`        | `1`               | How many may be in flight at once.                         |
| `maxRounds`          | `Infinity`        | Round ceiling.                                             |
| `maxReflectionCalls` | unbounded         | Bounded separately; no metric budget covers reflection.    |
| `historySize`        | `20`              | Scored attempts the prompt carries, strongest kept.        |
| `exemplars`          | `3`               | Task inputs shown for grounding.                           |
| `scoringSetSize`     | unset             | Instances drawn once from the trainingSet to screen on.    |
| `fullEvalInterval`   | `3`               | Rounds between full validationSet sweeps of the incumbent. |
| `scoreScale`         | `100`             | What scores are multiplied by before being shown.          |
| `seed`               | `0`               | Seeds the run's random stream.                             |
| `buildPrompt`        | `buildOproPrompt` | Replaces the meta-prompt template.                         |

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

`OproResult` adds `seedScore`, `rounds`, `trajectory` (every candidate scored, in order), `reflectionCalls`, and `cacheHits`. `stopReason` is `"budgetExhausted"`, `"reflectionBudgetExhausted"`, `"aborted"`, or `"maxRounds"`.

History is sorted by ascending score so the strongest attempt appears nearest the request. Scores are shown as integers because models distinguish 41 from 68 more reliably than 0.41 from 0.68.

For multi-component candidates, each attempt records the other components present when it was scored. The prompt only includes attempts from the current component context, avoiding comparisons between scores obtained with different companion text. Single-component candidates require no filtering.

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
| `datasetSummary`           | `true`             | Summarize the trainingSet once and show it to the proposer. One reflection call.                                      |
| `summaryExamples`          | `10`               | Trainset entries the summary is written from.                                                                         |
| `tips`                     | built-in           | Style hints, so a menu spreads over approaches.                                                                       |
| `buildPrompt`              | `buildMiproPrompt` | Replaces the proposal template.                                                                                       |

The proposer receives the current text of other components, a generated summary of `trainingSet`, and task exemplars. The summary uses one reflection call and is skipped when all menus are supplied through `componentOptions`. Unlike DSPy's MIPROv2 implementation, textopt does not include program source because the adapter interface has no generic representation for it.

`componentOptions` supplies menu entries **verbatim**, with no reflection call. The menu for a component is always its seed text followed by its options.

`demoComponents` identifies components that contain few-shot blocks. Their menus are bootstrapped from successful `trainingSet` rollouts instead of generated by `reflect`. Each demo set uses a separate shuffled pass and a size between one and `maxDemos`; the zero-shot option is always included. Separate passes can produce different demos for stochastic systems but add redundant work for deterministic systems, matching MIPROv2's behavior. Bootstrap rollouts count against `maxMetricCalls` and are also reported as `bootstrapMetricCalls`.

When `goldOutput` is supplied, each demo component also includes a labels-only `LabeledFewShot` option. It requires no rollout because the expected output is provided by the caller, and remains available when the seed system cannot produce successful bootstrap examples.

Every `fullEvalInterval` trials, the unswept configuration with the highest average minibatch score receives a full evaluation. A final sweep runs when the search ends. Averaging repeated minibatch scores reduces sensitivity to individual samples.

The surrogate includes both minibatch and full-evaluation observations, matching DSPy. The seed's full score is registered before the first trial. In a ten-seed sweep, adding promoted configurations' full scores reduced repeat proposals of disproved configurations from 21 to 10. Repeated high minibatch scores can still outweigh one low validation score.

One difference from MIPROv2 is that textopt keeps labelled demo sets as separate menu options; MIPROv2 pads bootstrapped sets with labelled examples.

`MiproResult` adds `seedScore`, `trials`, `menu` (the space actually searched), `observations`, `fullEvaluations`, `bootstrapMetricCalls`, `reflectionCalls`, and `cacheHits`. Only a full validation sweep can move the incumbent; minibatch readings decide what is worth sweeping.

**`proposeConfiguration({ observations, menuSizes, gamma, samples, startupTrials, multivariate, rng })`** is the surrogate on its own: it splits the observations into good and rest, models the density of each, samples from the good one and ranks the draws by the log ratio.

With `multivariate: true`, each density is a mixture of kernels centered on observed configurations. This preserves dependencies between component options. With `multivariate: false`, each component uses an independent smoothed histogram, which learns from fewer trials but cannot represent interactions.

In a measured space of 3,125 configurations with 60 trials, the joint model reached a mean best score of 0.87 and solved 8 of 15 runs; the independent model reached 0.78 and solved 5. In a 16-configuration space with 30 trials, the independent model found the optimum in 16 of 20 runs, compared with 14 for the joint model. The multivariate model is the default because it performs better when the space cannot be nearly enumerated.

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
| `seed`        | `0`                     | Seeds the run's random stream.                    |
| `buildPrompt` | `buildParaphrasePrompt` | Replaces the paraphrase template.                 |

`RandomSearchResult` adds `seedScore`, `rounds`, `variantsEvaluated`, `reflectionCalls`, and `cacheHits`.

The paraphrase prompt receives no score or feedback. Compare random search with a reflective optimizer under the same metric budget to measure the effect of reflection.

## `textopt/testing`

Deterministic keyword-coverage fixtures for testing optimizers and adapters without network access.

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
