# textopt

Framework-agnostic prompt optimization for TypeScript, with GEPA (Genetic-Pareto) as its first optimizer.

Zero dependencies. This is the API reference for the package. For what GEPA is and how the search works, see the [project README](../../README.md).

## Entry points

| Import            | Contains                                                                                                                                       |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `textopt`         | The optimizer and adapter contracts, the evaluation cache, and a concurrency helper. No engine: importing `GepaOptimizer` from here will fail. |
| `textopt/gepa`    | `GepaOptimizer` and everything GEPA-specific. Its config, task, result, adapter, events, snapshot, and the swappable strategies.               |
| `textopt/testing` | A deterministic, LLM-free system under optimization and reflection model, for exercising the loop or your own adapter without a network.       |

## `textopt`

```ts
import { componentNames, createMemoryCache, mapWithConcurrency } from "textopt";
import type {
  Adapter,
  BatchSampler,
  Candidate,
  CachedScore,
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

**`Optimizer<Stop extends string>`** has exactly one method, `optimize(task: OptimizerTask) => Promise<OptimizerResult>`. `OptimizerTask` is the run-level input every optimizer needs (`seedCandidate`, `trainset`, `valset`, `adapter`, `maxMetricCalls`, `signal`). `OptimizerResult` is what every optimizer reports (`bestCandidate`, `bestScore`, `bestOutputs`, `metricCalls`, `stopReason`). An optimizer's own task and result types extend these.

**`TextModel`** is `({ prompt, signal }) => Promise<string>`, the whole provider seam.

### Values

**`createMemoryCache({ maxEntries = 100_000, entries })`** returns an in-memory `EvaluationCache`. `entries` restores a previous run's checkpointed scores. Eviction is oldest-first once `maxEntries` is reached.

To back the cache with Redis, SQLite, or a file, implement **`EvaluationCache`** yourself: `get(key)`, `set(key, cached)`, and optionally `entries()`. Leave `entries()` off a store that is already durable and has nothing to checkpoint.

**`mapWithConcurrency({ items, limit, task, signal })`** is an order-preserving map with a concurrency limit. Adapters use it to fan a batch across a bounded number of in-flight model calls. Work already running settles before a failure propagates, so a rejected run does not keep spending budget in the background.

**`componentNames(candidate)`** is `Object.keys` that preserves the component union instead of widening it back to `string`.

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

| Option                     | Default                   | Effect                                                                     |
| -------------------------- | ------------------------- | -------------------------------------------------------------------------- |
| `minibatchSize`            | `3`                       | Instances per screening batch.                                             |
| `maxIterations`            | `Infinity`                | Iteration ceiling. The metric budget usually binds first.                  |
| `seed`                     | `0`                       | Seeds the run's random stream.                                             |
| `candidateSelector`        | `paretoSelector()`        | Which candidate becomes the next parent.                                   |
| `acceptance`               | `improvementAcceptance()` | Whether a screened child beats its parent.                                 |
| `merge.enabled`            | components > 1            | System-aware merge of two lineages.                                        |
| `merge.maxInvocations`     | `5`                       | Ceiling on merges attempted per run.                                       |
| `skipPerfectScore`         | `true`                    | Skip reflection when the parent already scores perfectly on the minibatch. |
| `perfectScore`             | `1`                       | The per-instance score treated as leaving no room.                         |
| `rejectedProposalMemory`   | `3`                       | Rejected texts per component shown back to reflection. `0` disables it.    |
| `proposals.perIteration`   | `1`                       | Mutations drawn per iteration, each with its own parent and minibatch.     |
| `proposals.concurrency`    | `1`                       | How many may be in flight at once.                                         |
| `proposals.selection`      | `"all"`                   | Which improving proposals to keep: `"all"`, `"best"`, or `{ keep: n }`.    |
| `reflection.maxCalls`      | unbounded                 | Hard ceiling on reflection calls. The run stops when it is reached.        |
| `reflection.maxRecords`    | unbounded                 | Records shown per component. The worst-scoring ones are kept.              |
| `reflection.maxCharacters` | unbounded                 | Rough ceiling on the serialized records.                                   |
| `reflection.buildPrompt`   | `buildReflectionPrompt`   | Replaces the prompt template. Custom proposers ignore it.                  |
| `checkpointCache`          | `true`                    | Include cached scores in every snapshot.                                   |
| `trackBestOutputs`         | `false`                   | Keep validation outputs so the winner's can be read back.                  |
| `raiseOnError`             | `true`                    | Rethrow adapter failures instead of skipping the iteration.                |

### `GepaTask`

One problem, its data, and its IO. Extends `OptimizerTask`.

Required: `seedCandidate`, `trainset`, `adapter` (a `GepaAdapter`), `reflect` (a `TextModel`), and `maxMetricCalls`.

| Option                | Default                                                                              |
| --------------------- | ------------------------------------------------------------------------------------ |
| `valset`              | the trainset                                                                         |
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

### Results, events, and resuming

**`GepaResult`** extends `OptimizerResult` with `bestCandidateId`, the full `candidates` pool (each a `CandidateRecord` with lineage, per-instance scores, and provenance), the `paretoFrontier`, the `scoreMatrix`, `perObjectiveBest`, `iterations`, `reflectionCalls`, `cacheHits`, and the final `snapshot`.

**`stopReason`** is one of `"budgetExhausted"`, `"reflectionBudgetExhausted"`, `"aborted"`, or `"maxIterations"`.

**`onEvent`** receives a discriminated `GepaEvent`: `start`, `iterationStart`, `evaluation`, `proposal`, `candidateAccepted`, `candidateRejected` (with `reason: "worse" | "notSelected"`), `error`, and `finish`.

**`onCheckpoint`** fires after the seed is scored and after every iteration, with a plain-JSON `GepaSnapshot`. Hand it back as `resumeFrom` to continue. Every snapshot is fingerprinted against its seed candidate, instance ids, and seed, so resuming against a different setup throws instead of silently scoring old candidates against new data.

## `textopt/testing`

A keyword-coverage task with a deterministic gradient, plus reflection models to match. No network, milliseconds per run.

| Export                       | What it is                                                                          |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `KEYWORD_EXAMPLES`           | Four `KeywordExample` rows.                                                         |
| `createKeywordAdapter()`     | A `GepaAdapter` scoring how many required terms the candidate text covers.          |
| `createKeywordReflector()`   | A `TextModel` that folds the missing terms into the current instruction.            |
| `createDegradingReflector()` | A `TextModel` that always proposes something strictly worse, for testing rejection. |
| `buildReflectionPrompt`      | The default prompt template, also exported from `textopt/gepa`.                     |

## Adapters

Prebuilt adapters for common stacks live alongside this package: [`@textopt/ai-sdk`](../ai-sdk), [`@textopt/langchain`](../langchain), and [`@textopt/braintrust`](../braintrust).

## License

MIT
