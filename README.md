# textopt

Framework-agnostic prompt optimization for TypeScript. Four search algorithms behind one contract, so the choice of optimizer is a line of code rather than a rewrite.

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

Swapping `GepaOptimizer` for `OproOptimizer`, `MiproOptimizer`, or `RandomSearchOptimizer` changes nothing else about the call.

## Status

Pre-release. The packages are `private` and not published to npm yet; use the workspace, or vendor the source. 400+ tests, none of which need a network.

## Choosing an optimizer

All four implement the same `Optimizer` interface, take the same adapter, and account for their budget the same way. What separates them is the signal they need and the structure they can exploit.

| Optimizer               | Signal it needs                   | Reach for it when                                                                             |
| ----------------------- | --------------------------------- | --------------------------------------------------------------------------------------------- |
| `GepaOptimizer`         | per-instance **textual feedback** | your metric can say _why_ a rollout failed, in words. Strongest of the four where that holds. |
| `OproOptimizer`         | a **scalar** score                | your metric is a number and nothing more: exact match, a pass rate, a latency budget.         |
| `MiproOptimizer`        | a **scalar** score                | components interact, and the right text for one depends on what the others say.               |
| `RandomSearchOptimizer` | a **scalar** score                | you want to know whether any of the above is earning its cost.                                |

The distinction that matters most is the first one. GEPA's advantage over ordinary prompt search is that it reads a paragraph explaining each failure and writes the next proposal from it — so a metric that emits only `0.0` gives its reflection step nothing to reason about, and the machinery goes on costing what it costs. That regime is what OPRO is for: it shows the model its own past attempts ordered by score and asks for one that scores higher, which needs no feedback at all.

MIPRO answers a different question. GEPA and OPRO both improve components one update at a time and screen each in isolation, which cannot see a pairing that only works as a pair. MIPRO builds a menu of candidate texts per component and searches over **joint configurations** with a Tree-structured Parzen Estimator, scoring most of them on cheap minibatches and promoting only the promising ones to a full sweep.

The joint part runs all the way down. Configurations are drawn, scored, and promoted as whole units, and the surrogate steering those draws models them as whole units too: its densities are mixtures of kernels centred on observed configurations rather than a histogram per component, so "B works only alongside A" survives into the next proposal instead of flattening into "B tends to appear in good trials". That is Optuna's multivariate sampler, which is what MIPROv2 turns on. Set `multivariate: false` for the independent model, which generalizes from fewer trials when the components really are separate.

`RandomSearchOptimizer` is deliberately the dumb one: it paraphrases components with no knowledge of how anything scored. It exists to be a control. Reflection costs frontier-model calls, and on an easy task an uninformed paraphrase sometimes matches it — the honest way to find out is to run both.

```ts
import { OproOptimizer } from "textopt/opro";
import { MiproOptimizer } from "textopt/mipro";
import { RandomSearchOptimizer } from "textopt/random-search";
```

## How GEPA works

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

| Package                 | What it is                                                                                                                                         |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `textopt`               | The optimizer and adapter contracts, the shared evaluator, demo bootstrapping, the evaluation cache, and the sampler/RNG types. Zero dependencies. |
| `textopt/gepa`          | `GepaOptimizer`, its config and types, and the pluggable selection/acceptance/evaluation strategies.                                               |
| `textopt/opro`          | `OproOptimizer`: score-history search for metrics that emit a number and no explanation.                                                           |
| `textopt/mipro`         | `MiproOptimizer`: joint search over per-component instruction menus, with a TPE surrogate and minibatch screening.                                 |
| `textopt/random-search` | `RandomSearchOptimizer`: uninformed paraphrase search, as the baseline the others have to beat.                                                    |
| `textopt/testing`       | A deterministic, LLM-free system under optimization and reflector, for exercising the loop and your own adapters in milliseconds.                  |
| `@textopt/ai-sdk`       | Adapter for the Vercel AI SDK (`generateText`, `generateObject`), with multi-step tool traces.                                                     |
| `@textopt/langchain`    | Adapter for any LangChain runnable, chain, agent, or LangGraph graph.                                                                              |
| `@textopt/braintrust`   | autoevals scorers as the metric, plus a decorator that logs every rollout to Braintrust.                                                           |

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

### Proposal strategies

By default every proposal is written by the same prompt: show the model the failures, ask for better text. Run that four times against one parent and you tend to get four versions of one idea, because the prompt frames the problem the same way each time.

`reflection.strategies` rotates over several framings instead, one per proposal slot:

```ts
import { diverseReflectionStrategies } from "textopt/gepa";

new GepaOptimizer({
  proposals: { perIteration: 4, concurrency: 4 },
  reflection: { strategies: diverseReflectionStrategies() },
});
```

The shipped rotation is the standard reflection prompt, plus one that **simplifies** (prompts accrete instructions as a run goes on, and the shortest version that still scores is usually the one that generalizes), one that **generalizes** away from the specific failures in the batch, and one that **rewrites** from scratch to break out of a lineage that has stopped moving. Each is exported on its own — `buildReflectionPrompt`, `buildSimplifyPrompt`, `buildGeneralizePrompt`, `buildRewritePrompt` — and any `ReflectionPromptBuilder` you write drops into the same list.

This is opt-in. The default remains the single published GEPA prompt.

## Few-shot demos

A demo block is just another component, so the search optimizes it like any other string. What is worth having is a way to fill it with examples the system actually got right, rather than hand-writing them.

**Before a run**, `bootstrapDemos` runs your seed candidate over the trainset and keeps the rollouts that scored well:

```ts
import { bootstrapDemos } from "textopt";

const { block, demos, metricCalls } = await bootstrapDemos({
  adapter,
  candidate: seedCandidate,
  trainset,
  minScore: 0.9,
  maxDemos: 4,
});

const seed = { instruction: "Route the ticket.", demos: block };
```

**During a run**, `createDemoProposer` harvests them from the reflective dataset the adapter already builds, which costs no extra rollouts and no reflection call:

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

A proposal appends to the block its parent already holds rather than replacing it, so examples accumulate along the accepted lineage — a demo persists only if the candidate carrying it beat its parent, the same bar every other component is held to.

## Configuring GEPA

`GepaConfig` (constructor) sets how the search runs, independent of your types:

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

`GepaTask` (per call) carries the problem, its data, and its IO: `seedCandidate`, `trainset`, `adapter`, `reflect`, `maxMetricCalls`, an optional `valset` that defaults to the trainset, an optional held-out `testset`, plus `componentSelector`, `batchSampler`, `valEvaluationPolicy`, `instanceId`, `cache`, `onEvent`, `onCheckpoint`, `resumeFrom`, `signal`.

Component names are inferred from `seedCandidate` and every other position is `NoInfer`, so a misspelled component is a compile error rather than a silent no-op.

Swappable strategies ship in `textopt/gepa`: `paretoSelector`, `currentBestSelector`, `epsilonGreedySelector`, `topKParetoSelector`, `roundRobinComponentSelector`, `allComponentsSelector`, `improvementAcceptance`, `fullEvaluationPolicy`, `subsampledEvaluationPolicy`.

## The other optimizers

Each takes the **base `Adapter`**, not `GepaAdapter`: they read scores only, so they never ask for traces or a reflective dataset. Every one reports `seedScore` alongside `bestScore`, so the lift the search actually bought is readable without arithmetic.

### OPRO

```ts
const result = await new OproOptimizer({
  proposalsPerRound: 4,
  historySize: 10,
  maxReflectionCalls: 40,
  seed: 11,
}).optimize({
  seedCandidate,
  trainset,
  valset,
  adapter,
  reflect,
  maxMetricCalls: 300,
});

result.trajectory; // every candidate scored, in the order it was tried
```

By default every proposal is scored on the whole `valset`. Set `scoringSetSize` and proposals are screened instead on a fixed slice of the trainset, with the incumbent swept in full every `fullEvalInterval` rounds — the paper's economics, and what makes a valset large enough to trust affordable. On a 30-instance valset, screening on 12 halved the rollouts and cost nothing in quality.

The meta-prompt carries the strongest attempts so far with their scores, in **ascending** order, and asks for one that scores higher. The ordering is load-bearing rather than cosmetic — the best attempt sits closest to the request, where the model attends to it most. Scores are scaled to integers (`scoreScale`, default 100) for the same reason: models discriminate 41 from 68 far more reliably than 0.41 from 0.68.

### MIPRO

```ts
const result = await new MiproOptimizer({
  instructionsPerComponent: 5,
  minibatchSize: 5,
  maxTrials: 30,
  seed: 11,
}).optimize({
  seedCandidate,
  trainset,
  valset,
  adapter,
  reflect,
  demoComponents: ["demos"], // menu bootstrapped from the trainset
  maxMetricCalls: 600,
});

result.menu; // the space that was searched, per component
result.observations; // every configuration tried, and which earned a full sweep
```

It first builds a menu per component — the seed text, plus instructions written by `reflect` against varied style hints so the menu spreads over approaches instead of rewording one idea. Then it searches configurations of that menu with a TPE surrogate: model where the good trials live versus the rest, sample from the good density, take the best of the batch. Trials are scored on cheap minibatches, and only the configurations that screen well there are promoted to a full validation sweep, so the expensive measurement is spent on the candidates that earned it.

`componentOptions` supplies menu entries verbatim, with no reflection call. `demoComponents` goes further and bootstraps a component's menu from the trainset, which is MIPROv2's other half — instructions and demonstrations searched together rather than instructions alone. Demos are harvested, not authored, so no reflection model ever sees them.

Promotion follows MIPROv2's cadence rather than a running bar: every `fullEvalInterval` trials, the configuration with the best **average** minibatch reading that has not been swept yet earns a full evaluation. Averaging matters — a single minibatch is a noisy reading, and promoting on one alone lets a lucky draw decide the run.

### Random search

```ts
const result = await new RandomSearchOptimizer({
  variants: 4,
  seed: 11,
}).optimize({
  seedCandidate,
  trainset,
  valset,
  adapter,
  reflect,
  maxMetricCalls: 300,
});
```

It paraphrases one component per round and keeps whatever scores best. The paraphrase prompt states outright that it has no information about how the current text performed — the ablation is the point. Compare its `bestScore` against a reflective run on the same budget: that difference is what reflection bought on _your_ task, and it is the only way to know whether it was worth the frontier-model calls.

## Measuring honestly

Selection pressure is applied to the valset for the entire run: candidates are kept because they win on it, and the winner is the one that won there most. So `bestScore` is fitted to the valset by construction, and reporting it as the improvement overstates the improvement by an amount nobody can see from the number itself.

Pass a `testset` and the winner is scored on it once, after the search is over:

```ts
const result = await optimizer.optimize({
  seedCandidate,
  trainset,
  valset,
  testset, // never seen by the search
  adapter,
  reflect,
  maxMetricCalls: 300,
});

result.bestScore; // on the valset — the search selected for this
result.testScore; // on instances no candidate was ever selected against
result.testMetricCalls; // charged separately, not against maxMetricCalls
```

A gap between the two is the overfitting, quantified. The held-out sweep is measurement rather than search, so it does not come out of `maxMetricCalls` and is reported on its own; and the resume fingerprint deliberately ignores the testset, so adding one to an existing run is not treated as a different problem.

Every optimizer supports this — it is on the shared `OptimizerTask` and `OptimizerResult`, not on GEPA.

## Budget, caching, resume

- **Caching.** Scores are cached per split + candidate text + instance id, and hits are not charged to the budget: a sweep is priced against the cache before it runs. The key covers the _whole_ candidate, so a child never reuses its parent's scores. Hits come from re-scoring the same candidate. The common case is a merged candidate screened on a validation subsample and then swept in full, along with partial `valEvaluationPolicy` sweeps and resumed runs replaying against a checkpointed cache. Instance ids default to a content hash of the datum, falling back to its position when it will not serialize, so pass `instanceId` for non-serializable data or for ids you can read in a trace. Pass `cache: false` to disable.
- **Checkpoints.** `onCheckpoint` fires after the seed is scored and after every iteration with a plain-JSON `GepaSnapshot`: the candidate pool, budget spent, RNG position, sampler state, rejected proposals, merge bookkeeping, and by default the cached scores. Hand it back as `resumeFrom` and the run continues on the same trajectory it would have taken uninterrupted. Every snapshot is fingerprinted against its seed candidate, instance ids, and seed, so resuming against a different setup is refused rather than silently mis-scored.
- **Events.** `onEvent` emits a typed stream: `start`, `iterationStart`, `evaluation`, `proposal`, `candidateAccepted`, `candidateRejected` (with why), `error`, `finish`.
- **Result.** Beyond `bestCandidate` / `bestScore`: the held-out `testScore` and `testMetricCalls` when a testset was given, the full `candidates` pool with lineage, the `paretoFrontier`, the `scoreMatrix`, `perObjectiveBest`, `metricCalls`, `reflectionCalls`, `cacheHits`, `stopReason`, and the final `snapshot`.

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
