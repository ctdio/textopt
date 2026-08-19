# textopt

Prompt optimization for TypeScript, with GEPA, OPRO, MIPRO, and random search behind a shared interface.

A run takes a seed candidate, a dataset, and an adapter that evaluates each rollout. Reflective optimizers use the adapter's textual feedback to propose better candidates.

```ts
const result = await new GepaOptimizer({ minibatchSize: 3, seed: 11 }).optimize(
  {
    seedCandidate: {
      system: "Classify the support ticket. Answer with one word.",
    },
    trainingSet,
    validationSet,
    adapter, // how to run and score your system
    reflect, // any text-in, text-out model
    maxMetricCalls: 150,
  },
);

result.bestCandidate; // { system: "..." }, same keys as the seed, checked at compile time
```

A candidate is a record of named strings. Components can be system prompts, tool descriptions, routing rules, few-shot blocks, regexes, or configuration. All four optimizers use the same call shape.

## Status

Pre-release. The packages are private and not yet published to npm. The test suite has more than 400 tests and runs without network access.

## Choosing an optimizer

All four use the same `Optimizer` interface, adapter, and budget accounting.

| Optimizer               | Required signal                   | Use it for                                              |
| ----------------------- | --------------------------------- | ------------------------------------------------------- |
| `GepaOptimizer`         | per-instance **textual feedback** | revising text from written explanations of failures     |
| `OproOptimizer`         | a **scalar** score                | proposing text from a history of scored attempts        |
| `MiproOptimizer`        | a **scalar** score                | searching combinations of interacting component options |
| `RandomSearchOptimizer` | a **scalar** score                | establishing a score-independent paraphrasing baseline  |

Use GEPA when the metric can explain failures in text. A scalar such as `0.0` gives its reflection step little useful information. OPRO only needs scalar scores: its prompt lists previous attempts by score and asks the model to improve on them.

GEPA and OPRO update components separately. MIPRO instead searches combinations of per-component options, which lets it find options that work well only together. It screens configurations on minibatches and evaluates promising ones against the full validation set.

MIPRO's default multivariate TPE models complete configurations. Set `multivariate: false` to model each component independently; that usually needs fewer observations but cannot model interactions between components.

`RandomSearchOptimizer` paraphrases components without using their scores. Run it as a baseline to check whether reflection improves enough to justify its model calls.

```ts
import { OproOptimizer } from "textopt/opro";
import { MiproOptimizer } from "textopt/mipro";
import { RandomSearchOptimizer } from "textopt/random-search";
```

## How GEPA works

GEPA maintains a pool of candidates and a **Pareto frontier over validation instances**, not objectives. A candidate remains on the frontier while it has the best score for at least one instance. Parent sampling is weighted by the number of instances each candidate wins, preserving candidates with useful strengths even when their mean score is lower.

Each iteration:

1. **Select** a parent from the frontier, and one or more of its components to update.
2. **Evaluate** the parent on a fresh minibatch, capturing traces.
3. **Reflect** on per-component evidence from the scored batch: inputs, outputs, feedback, and scores. Recent rejected proposals are included to reduce repetition.
4. **Screen** the child on the same minibatch. Only improvements proceed.
5. **Sweep** accepted children over the validation set and update the frontier.
6. **Merge** lineages that improved different components. Merge is enabled by default for multi-component candidates.

`maxMetricCalls` limits scored rollouts; cache hits do not count. Reflection calls have a separate limit.

Based on _GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning_.

## Packages

| Package                 | Contents                                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------------------- |
| `textopt`               | Shared contracts, evaluator, demo utilities, cache, sampler, and RNG types. No runtime dependencies. |
| `textopt/gepa`          | `GepaOptimizer`, GEPA types, and configurable selection, acceptance, and evaluation strategies.      |
| `textopt/opro`          | `OproOptimizer` and score-history prompting.                                                         |
| `textopt/mipro`         | `MiproOptimizer`, per-component option menus, TPE search, and minibatch screening.                   |
| `textopt/random-search` | `RandomSearchOptimizer`, a score-independent paraphrasing baseline.                                  |
| `textopt/testing`       | Deterministic fixtures for testing optimizers and adapters without an LLM.                           |
| `@textopt/ai-sdk`       | Vercel AI SDK adapter for `generateText` and `generateObject`, including multi-step tool traces.     |
| `@textopt/langchain`    | Adapter for LangChain runnables, chains, agents, and LangGraph graphs.                               |
| `@textopt/braintrust`   | autoevals scorer integration and a Braintrust logging decorator.                                     |

## Quickstart, offline

No API keys, no network:

```ts
import { GepaOptimizer } from "textopt/gepa";
import {
  KEYWORD_EXAMPLES,
  createKeywordAdapter,
  createKeywordReflector,
} from "textopt/testing";

const gepa = new GepaOptimizer({ minibatchSize: 2, seed: 7 });

const result = await gepa.optimize({
  seedCandidate: { instruction: "Answer the customer's question." },
  trainingSet: KEYWORD_EXAMPLES,
  adapter: createKeywordAdapter(),
  reflect: createKeywordReflector(),
  maxMetricCalls: 120,
  onEvent: (event) => {
    if (event.type === "candidateAccepted") {
      console.log(
        `accepted #${event.candidateId} score=${event.aggregateScore}`,
      );
    }
  },
});

console.log(result.bestScore, result.bestCandidate.instruction);
```

Constructor options control search behavior. `optimize` receives the dataset, adapter, candidate, and run budget. Optimizer instances do not retain state between runs.

## The adapter

The adapter connects an optimizer to the system being evaluated:

```ts
interface GepaAdapter<Datum, Trajectory, Output, K extends string> {
  evaluate(
    args: EvaluateArgs<Datum, K>,
  ): Promise<EvaluationBatch<Trajectory, Output>>;
  makeReflectiveDataset(
    args: MakeReflectiveDatasetArgs<Datum, Trajectory, Output, K>,
  ): ReflectiveDataset<K>;
  proposeNewTexts?(args: ProposeArgs<K>): ComponentPatch<K>; // replaces the reflection LLM entirely
}
```

Methods may be synchronous or asynchronous; the example shows the asynchronous form.

`evaluate` returns one score per instance and may include textual feedback. GEPA uses that feedback during reflection.

- **`args.run`** identifies the rollout's `iteration`, `phase`, `split`, and `candidateId`. Forward it to your tracing system.
- **`transient`** marks scores caused by infrastructure failures such as rate limits or 5xx responses. Transient scores are not cached.

### Vercel AI SDK

```ts
import { createAiSdkAdapter } from "@textopt/ai-sdk";
import { generateText } from "ai";

const adapter = createAiSdkAdapter<Ticket>({
  run: ({ candidate, datum, signal }) =>
    generateText({
      model: taskModel,
      system: candidate.system ?? "",
      prompt: datum.text,
      abortSignal: signal,
    }),

  score: ({ datum, output }) =>
    output === datum.label
      ? { score: 1, feedback: `Correct: ${datum.label}.` }
      : {
          score: 0,
          feedback: `Predicted "${output}" but the correct queue is "${datum.label}". ${datum.why}`,
        },

  concurrency: 4,
});
```

### LangChain

```ts
import { createLangChainAdapter } from "@textopt/langchain";

const adapter = createLangChainAdapter<Ticket, string>({
  buildRunnable: (candidate) => buildChain(candidate.system, candidate.rubric),
  score: ({ datum, output, trace }) => ({
    score: grade(datum, output),
    feedback: explain(trace),
  }),
});
```

The adapter rebuilds the runnable for each candidate. Traces include LLM, tool, and retriever spans; set `includeChainSteps` to include chain spans. Each rollout also includes `textopt_iteration`, `textopt_phase`, `textopt_split`, and `textopt_candidate_id` metadata for filtering in LangSmith.

### Braintrust

```ts
import {
  createBraintrustScorer,
  withBraintrustLogging,
} from "@textopt/braintrust";
import { ExactMatch, Levenshtein } from "autoevals";

const score = createBraintrustScorer<string>({
  scorers: [ExactMatch, Levenshtein],
  weights: { ExactMatch: 3 },
});

const adapter = withBraintrustLogging({
  adapter: baseAdapter,
  logger: initLogger({ projectName: "ticket-routing" }),
});
```

The scorer maps scorer rationales to feedback and individual scores to `objectiveScores`. The logging decorator works with the AI SDK adapter, the LangChain adapter, or a custom adapter.

## The reflection model

`reflect` implements the provider-independent `TextModel` interface: `({ prompt, signal }) => Promise<string>`.

```ts
import type { TextModel } from "textopt";
import { generateText } from "ai";

const reflect: TextModel = async ({ prompt, signal }) => {
  const result = await generateText({ model, prompt, abortSignal: signal });
  return result.text;
};
```

A LangChain chat model, vendor SDK call, local model, or deterministic function can implement `TextModel`. If the adapter implements `proposeNewTexts`, it generates proposals without calling `reflect`; the type still requires `reflect`, so pass a stub as shown in the `pareto` example.

The model under optimization is usually cheaper than the reflection model, which must analyze failures and revise the candidate.

### Proposal strategies

By default, every proposal uses the same reflection prompt. Multiple calls against one parent can therefore produce similar revisions.

`reflection.strategies` rotates over several framings instead, one per proposal slot:

```ts
import { diverseReflectionStrategies } from "textopt/gepa";

new GepaOptimizer({
  proposals: { perIteration: 4, concurrency: 4 },
  reflection: { strategies: diverseReflectionStrategies() },
});
```

The included rotation alternates the standard prompt with prompts that simplify, generalize, or rewrite the candidate. The builders are exported as `buildReflectionPrompt`, `buildSimplifyPrompt`, `buildGeneralizePrompt`, and `buildRewritePrompt`; custom `ReflectionPromptBuilder` functions use the same interface.

This behavior is opt-in. The default is the prompt from the GEPA reference implementation.

## Few-shot demos

Few-shot examples can be stored in a candidate component and optimized with the other text. textopt can populate that component from successful training rollouts.

Before optimization, `bootstrapDemos` evaluates the seed candidate on `trainingSet` and keeps high-scoring rollouts:

```ts
import { bootstrapDemos } from "textopt";

const { block, demos, metricCalls } = await bootstrapDemos({
  adapter,
  candidate: seedCandidate,
  trainingSet,
  minScore: 0.9,
  maxDemos: 4,
});

const seed = { instruction: "Route the ticket.", demos: block };
```

During optimization, `createDemoProposer` reads demos from the existing reflective dataset without additional rollouts or reflection calls:

```ts
import { createDemoProposer } from "textopt/gepa";

const adapter = {
  ...baseAdapter,
  proposeNewTexts: createDemoProposer({
    components: ["demos"],
    minScore: 0.9,
    maxDemos: 4,
    fallback: baseProposer, // writes the other components normally
  }),
};
```

Each proposal appends demos to its parent's block. A demo remains in the lineage only when the candidate containing it is accepted.

## Configuring GEPA

Pass search settings to the `GepaOptimizer` constructor:

```ts
new GepaOptimizer({
  minibatchSize: 3,
  maxIterations: 50,
  seed: 11,
  proposals: { perIteration: 3, concurrency: 3, selection: "best" },
  reflection: { maxCalls: 40, maxRecords: 5, maxCharacters: 20_000 },
  // ...or `strategies: diverseReflectionStrategies()` in place of one prompt
  merge: { enabled: true, maxInvocations: 5 },
  candidateSelector: paretoSelector(),
  acceptance: improvementAcceptance(),
  skipPerfectScore: true,
  rejectedProposalMemory: 3,
  trackBestOutputs: true,
});
```

`optimize` takes a `GepaTask`. Required fields are `seedCandidate`, `trainingSet`, `adapter`, `reflect`, and `maxMetricCalls`. `validationSet` defaults to `trainingSet`; `testSet` is optional and held out. Other options are `componentSelector`, `batchSampler`, `valEvaluationPolicy`, `instanceId`, `cache`, `onEvent`, `onCheckpoint`, `resumeFrom`, and `signal`.

Component names are inferred from `seedCandidate`. Other positions use `NoInfer`, so misspelled component names fail type checking.

`textopt/gepa` exports the following strategies: `paretoSelector`, `currentBestSelector`, `epsilonGreedySelector`, `topKParetoSelector`, `roundRobinComponentSelector`, `allComponentsSelector`, `improvementAcceptance`, `fullEvaluationPolicy`, and `subsampledEvaluationPolicy`.

## The other optimizers

These optimizers use the base `Adapter` because they do not require GEPA's reflective dataset. Their results include both `seedScore` and `bestScore`.

### OPRO

```ts
const result = await new OproOptimizer({
  proposalsPerRound: 4,
  historySize: 10,
  maxReflectionCalls: 40,
  seed: 11,
}).optimize({
  seedCandidate,
  trainingSet,
  validationSet,
  adapter,
  reflect,
  maxMetricCalls: 300,
});

result.trajectory; // every candidate scored, in the order it was tried
```

By default, every proposal is scored on the full `validationSet`. With `scoringSetSize`, proposals are screened on a fixed subset of `trainingSet`, and the incumbent receives a full sweep every `fullEvalInterval` rounds. In a 30-instance validation set, screening on 12 instances halved rollout count without reducing the measured best score.

The meta-prompt lists the strongest attempts in ascending score order, placing the best attempt nearest the request. `scoreScale` converts scores to integers (100 by default), because models distinguish values such as 41 and 68 more reliably than 0.41 and 0.68.

### MIPRO

```ts
const result = await new MiproOptimizer({
  instructionsPerComponent: 5,
  minibatchSize: 5,
  maxTrials: 30,
  seed: 11,
}).optimize({
  seedCandidate,
  trainingSet,
  validationSet,
  adapter,
  reflect,
  demoComponents: ["demos"], // menu bootstrapped from the training set
  maxMetricCalls: 600,
});

result.menu; // the space that was searched, per component
result.observations; // every configuration tried, and which earned a full sweep
```

MIPRO first builds a menu for each component from the seed text and variants generated by `reflect`. A TPE surrogate then proposes configurations from those menus. Trials run on minibatches; selected configurations receive a full validation sweep.

`componentOptions` adds menu entries without reflection calls. `demoComponents` builds menus of few-shot blocks from successful training rollouts, allowing instructions and demonstrations to be searched together.

Every `fullEvalInterval` trials, MIPRO fully evaluates the unswept configuration with the highest average minibatch score. Averaging repeated observations reduces the effect of a lucky minibatch.

### Random search

```ts
const result = await new RandomSearchOptimizer({
  variants: 4,
  seed: 11,
}).optimize({
  seedCandidate,
  trainingSet,
  validationSet,
  adapter,
  reflect,
  maxMetricCalls: 300,
});
```

Random search paraphrases one component per round and keeps the highest-scoring candidate. Its prompt receives no performance data. Compare it with a reflective optimizer under the same metric budget to measure the benefit of reflection.

## Held-out evaluation

The optimizer selects candidates against `validationSet`, so `bestScore` is fitted to that set and may overstate performance on unseen data.

Pass a `testSet` and the winner is scored on it once, after the search is over:

```ts
const result = await optimizer.optimize({
  seedCandidate,
  trainingSet,
  validationSet,
  testSet, // never seen by the search
  adapter,
  reflect,
  maxMetricCalls: 300,
});

result.bestScore; // on the validation set — the search selected for this
result.testScore; // on instances no candidate was ever selected against
result.testMetricCalls; // charged separately, not against maxMetricCalls
```

The gap between `bestScore` and `testScore` estimates validation overfitting. Test rollouts are reported separately and do not count against `maxMetricCalls`. The resume fingerprint ignores `testSet`, so it can be added when resuming a run.

All optimizers expose these fields through `OptimizerTask` and `OptimizerResult`.

## Budget, caching, resume

- **Caching.** Cache keys include the split, complete candidate, and instance ID. Cache hits do not count against the metric budget. Instance IDs default to a content hash, with the row position as a fallback for values that cannot be serialized. Provide `instanceId` for non-serializable data or readable trace IDs. Set `cache: false` to disable caching.
- **Checkpoints.** `onCheckpoint` runs after seed evaluation and each iteration. Its `GepaSnapshot` contains the candidate pool, budgets, RNG and sampler state, rejected proposals, merge state, and cached scores. Resume with `resumeFrom`. A fingerprint prevents resuming with a different seed candidate, instance set, or random seed.
- **Events.** `onEvent` receives `start`, `iterationStart`, `evaluation`, `proposal`, `candidateAccepted`, `candidateRejected`, `error`, and `finish` events.
- **Result.** The result includes scores, held-out test results, candidate lineage, the Pareto frontier, score matrix, per-objective best candidates, call counts, stop reason, and final snapshot.

## Examples

Runnable scripts in [`examples/src`](examples/src):

| Command                                     | Needs               | Demonstrates                                                                                                 |
| ------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `pnpm --filter textopt-examples keyword`    | none                | An offline optimization run.                                                                                 |
| `pnpm --filter textopt-examples pareto`     | none                | The instance-level Pareto frontier.                                                                          |
| `pnpm --filter textopt-examples ai-sdk`     | `ANTHROPIC_API_KEY` | Optimization of one `generateText` call.                                                                     |
| `pnpm --filter textopt-examples langchain`  | `OPENAI_API_KEY`    | Optimization of a LangChain chain.                                                                           |
| `pnpm --filter textopt-examples braintrust` | `OPENAI_API_KEY`    | autoevals scoring and Braintrust logging. Without `BRAINTRUST_API_KEY`, the script prints events locally.    |
| `pnpm --filter textopt-examples merge`      | `ANTHROPIC_API_KEY` | System-aware merging of two components from separate lineages.                                               |
| `pnpm --filter textopt-examples custom`     | `ANTHROPIC_API_KEY` | A custom adapter over a vendor SDK. Set `VENDOR` to use OpenAI instead; this also requires `OPENAI_API_KEY`. |

Keys are read from a root `.env` if present.

## Development

Requires Node >= 20 and pnpm 10.

```bash
pnpm install
pnpm test          # vitest, runs against source via workspace aliases
pnpm build         # tsdown, per package
pnpm typecheck     # build, then tsc --noEmit everywhere
```

## License

MIT
