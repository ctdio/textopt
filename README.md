# textopt

Framework-agnostic prompt optimization for TypeScript, with GEPA (Genetic-Pareto) as its first optimizer.

You give it a seed candidate (a map of named text components), a dataset, and a way to score a rollout with _textual feedback_. It searches for text that scores better, reading the feedback to write each proposal instead of mutating blindly.

```ts
const result = await new GepaOptimizer({ minibatchSize: 3, seed: 11 }).optimize(
  {
    seedCandidate: {
      system: "Classify the support ticket. Answer with one word.",
    },
    trainset,
    valset,
    adapter, // how to run and score your system
    reflect, // any text-in, text-out model
    maxMetricCalls: 150,
  },
);

result.bestCandidate; // { system: "..." }, same keys as the seed, checked at compile time
```

Nothing about the loop is prompt-specific. A candidate component is any named string the system reads: a system prompt, a tool description, a routing rule, a few-shot block, a regex, a config blob.

## Status

Pre-release. The packages are `private` and not published to npm yet; use the workspace, or vendor the source. 300+ tests, none of which need a network.

## How it works

GEPA keeps a pool of candidates and a **Pareto frontier taken over validation instances**, not over objectives. `scoreMatrix[candidate][instance]` is the whole selection state: a candidate survives if it is best on at least one instance, even when its mean is mediocre, and parents are sampled in proportion to how many instances they win. That is what stops the search collapsing onto a single local optimum.

Each iteration:

1. **Select** a parent from the frontier, and one or more of its components to update.
2. **Evaluate** the parent on a fresh minibatch, capturing traces.
3. **Reflect**: the adapter turns the scored batch into per-component evidence (inputs, outputs, feedback, scores), and a reflection model writes new text from it. Proposals already tried and rejected get shown back, so the run stops re-deriving dead ends.
4. **Screen** the child on the same minibatch. It has to beat its parent there before it costs anything more.
5. **Sweep** the survivors over the validation set and record them on the frontier.
6. **Merge**, periodically: when two lineages improved different components, system-aware merge recombines them without re-running the search. Enabled by default for multi-component candidates.

The budget is denominated in metric calls (`maxMetricCalls`); cached rollouts are free. Reflection is bounded separately, since it is often the expensive part and no metric budget covers it.

Based on _GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning_.

## Packages

| Package               | What it is                                                                                                                        |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `textopt`             | The optimizer and adapter contracts, the evaluation cache, a concurrency helper, and the sampler/RNG types. Zero dependencies.    |
| `textopt/gepa`        | `GepaOptimizer`, its config and types, and the pluggable selection/acceptance/evaluation strategies.                              |
| `textopt/testing`     | A deterministic, LLM-free system under optimization and reflector, for exercising the loop and your own adapters in milliseconds. |
| `@textopt/ai-sdk`     | Adapter for the Vercel AI SDK (`generateText`, `generateObject`), with multi-step tool traces.                                    |
| `@textopt/langchain`  | Adapter for any LangChain runnable, chain, agent, or LangGraph graph.                                                             |
| `@textopt/braintrust` | autoevals scorers as the metric, plus a decorator that logs every rollout to Braintrust.                                          |

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
  trainset: KEYWORD_EXAMPLES,
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

The constructor takes settings that are stateless and free of your types; `optimize` takes one problem. An optimizer holds no run state, so one instance can run any number of them.

## The adapter

The adapter is the only integration seam. Two methods make any system optimizable:

```ts
interface GepaAdapter<Datum, Traj, Out, K extends string> {
  evaluate(args: EvaluateArgs<Datum, K>): Promise<EvaluationBatch<Traj, Out>>;
  makeReflectiveDataset(
    args: MakeReflectiveDatasetArgs<Datum, Traj, Out, K>,
  ): ReflectiveDataset<K>;
  proposeNewTexts?(args: ProposeArgs<K>): ComponentPatch<K>; // replaces the reflection LLM entirely
}
```

Every method may be sync or async; the signatures above are shown in one form for brevity.

`evaluate` returns one score per instance plus, ideally, one paragraph of feedback per instance. A number tells the search how wrong a candidate was. The feedback tells the reflection model what was wrong, which is the difference between GEPA and random prompt search.

Two things worth wiring up:

- **`args.run`** says where a rollout sits in the optimization (`iteration`, `phase`, `split`, `candidateId`). Forward it to whatever tracing you already have, or your traces are thousands of indistinguishable calls.
- **`transient`** marks a score produced by a rate limit or a 5xx rather than by the candidate. Transient scores are never cached, so an outage doesn't pin a good candidate to a permanent zero.

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

The candidate is injected by rebuilding the runnable, so anything downstream that reads candidate text is optimizable. LLM, tool, and retriever spans land in the trace (chain spans too, behind `includeChainSteps`), and every rollout carries `textopt_iteration` / `textopt_phase` / `textopt_split` / `textopt_candidate_id` metadata, so a LangSmith project can be filtered down to the iteration whose score moved.

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

The scorer carries each scorer's rationale through as feedback and each scorer's number through as `objectiveScores`. Logging is a decorator, so it composes with the AI SDK adapter, the LangChain one, or your own.

## The reflection model

`reflect` is the other required input, and the interface is small: `({ prompt, signal }) => Promise<string>`. Text in, text out, provider-agnostic.

```ts
import type { TextModel } from "textopt";
import { generateText } from "ai";

const reflect: TextModel = async ({ prompt, signal }) => {
  const result = await generateText({ model, prompt, abortSignal: signal });
  return result.text;
};
```

A LangChain chat model, a raw vendor SDK call, a local model, or a hand-written rule are each an equally valid `TextModel`. An adapter that implements `proposeNewTexts` writes proposals itself and never calls this one, though `reflect` is still required by the type, so pass a stub the way the `pareto` example does.

One choice worth making on purpose: the system under optimization wants the cheap model, since it is the thing being made better, while reflection wants a frontier model, since it is the part doing the reasoning about failure.

## Configuration

`GepaConfig` (constructor) sets how the search runs, independent of your types:

```ts
new GepaOptimizer({
  minibatchSize: 3,
  maxIterations: 50,
  seed: 11,
  proposals: { perIteration: 3, concurrency: 3, selection: "best" },
  reflection: { maxCalls: 40, maxRecords: 5, maxCharacters: 20_000 },
  merge: { enabled: true, maxInvocations: 5 },
  candidateSelector: paretoSelector(),
  acceptance: improvementAcceptance(),
  skipPerfectScore: true,
  rejectedProposalMemory: 3,
  trackBestOutputs: true,
});
```

`GepaTask` (per call) carries the problem, its data, and its IO: `seedCandidate`, `trainset`, `adapter`, `reflect`, `maxMetricCalls`, an optional `valset` that defaults to the trainset, plus `componentSelector`, `batchSampler`, `valEvaluationPolicy`, `instanceId`, `cache`, `onEvent`, `onCheckpoint`, `resumeFrom`, `signal`.

Component names are inferred from `seedCandidate` and every other position is `NoInfer`, so a misspelled component is a compile error rather than a silent no-op.

Swappable strategies ship in `textopt/gepa`: `paretoSelector`, `currentBestSelector`, `epsilonGreedySelector`, `topKParetoSelector`, `roundRobinComponentSelector`, `allComponentsSelector`, `improvementAcceptance`, `fullEvaluationPolicy`, `subsampledEvaluationPolicy`.

## Budget, caching, resume

- **Caching.** Scores are cached per split + candidate text + instance id, and hits are not charged to the budget: a sweep is priced against the cache before it runs. The key covers the _whole_ candidate, so a child never reuses its parent's scores. Hits come from re-scoring the same candidate. The common case is a merged candidate screened on a validation subsample and then swept in full, along with partial `valEvaluationPolicy` sweeps and resumed runs replaying against a checkpointed cache. Instance ids default to a content hash of the datum, falling back to its position when it will not serialize, so pass `instanceId` for non-serializable data or for ids you can read in a trace. Pass `cache: false` to disable.
- **Checkpoints.** `onCheckpoint` fires after the seed is scored and after every iteration with a plain-JSON `GepaSnapshot`: the candidate pool, budget spent, RNG position, sampler state, rejected proposals, merge bookkeeping, and by default the cached scores. Hand it back as `resumeFrom` and the run continues on the same trajectory it would have taken uninterrupted. Every snapshot is fingerprinted against its seed candidate, instance ids, and seed, so resuming against a different setup is refused rather than silently mis-scored.
- **Events.** `onEvent` emits a typed stream: `start`, `iterationStart`, `evaluation`, `proposal`, `candidateAccepted`, `candidateRejected` (with why), `error`, `finish`.
- **Result.** Beyond `bestCandidate` / `bestScore`: the full `candidates` pool with lineage, the `paretoFrontier`, the `scoreMatrix`, `perObjectiveBest`, `metricCalls`, `reflectionCalls`, `cacheHits`, `stopReason`, and the final `snapshot`.

## Examples

Runnable scripts in [`examples/src`](examples/src):

| Command                                     | Needs               | What it shows                                                                                                                                        |
| ------------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pnpm --filter textopt-examples keyword`    | none                | The whole loop end to end, offline.                                                                                                                  |
| `pnpm --filter textopt-examples pareto`     | none                | What the instance-level frontier is, printed. Offline.                                                                                               |
| `pnpm --filter textopt-examples ai-sdk`     | `ANTHROPIC_API_KEY` | Optimizing one `generateText` call.                                                                                                                  |
| `pnpm --filter textopt-examples langchain`  | `OPENAI_API_KEY`    | Optimizing a LangChain chain.                                                                                                                        |
| `pnpm --filter textopt-examples braintrust` | `OPENAI_API_KEY`    | autoevals as the metric, rollouts logged as experiment rows. `BRAINTRUST_API_KEY` is optional; without it the events are printed instead of shipped. |
| `pnpm --filter textopt-examples merge`      | `ANTHROPIC_API_KEY` | Two components, two lineages, system-aware merge.                                                                                                    |
| `pnpm --filter textopt-examples custom`     | `ANTHROPIC_API_KEY` | A hand-written adapter over a vendor SDK, no framework. Flip the `VENDOR` constant to run it on OpenAI instead, which needs `OPENAI_API_KEY`.        |

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
