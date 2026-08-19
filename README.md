# textopt

Prompt optimization for TypeScript, with GEPA, SIMBA, OPRO, MIPRO, bootstrapped few-shot search, and random search behind a shared interface.

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

A candidate is a record of named strings. Components can be system prompts, tool descriptions, routing rules, few-shot blocks, regexes, or configuration. Every optimizer uses the same call shape.

## Status

`textopt` and `@textopt/langchain` are the packages this repository releases. The test suite has more than 550 tests and runs without network access.

`@textopt/ai-sdk` and `@textopt/braintrust` are beta and are not published to npm. Use them from a checkout of this repository until their interfaces settle.

## Choosing an optimizer

All six use the same `Optimizer` interface, adapter, and budget accounting.

| Optimizer                  | Required signal                   | Use it for                                                        |
| -------------------------- | --------------------------------- | ----------------------------------------------------------------- |
| `GepaOptimizer`            | per-instance **textual feedback** | revising text from written explanations of failures               |
| `SimbaOptimizer`           | per-instance **textual feedback** | noisy metrics, where the same instance scores differently per run |
| `OproOptimizer`            | a **scalar** score                | proposing text from a history of scored attempts                  |
| `MiproOptimizer`           | a **scalar** score                | searching combinations of interacting component options           |
| `BootstrapSearchOptimizer` | a **scalar** score                | few-shot demonstrations, with no proposal model at all            |
| `RandomSearchOptimizer`    | a **scalar** score                | establishing a score-independent paraphrasing baseline            |

Use GEPA when the metric can explain failures in text. A scalar such as `0.0` gives its reflection step little useful information. OPRO only needs scalar scores: its prompt lists previous attempts by score and asks the model to improve on them.

SIMBA also needs textual feedback, but reads a different signal. Instead of reflecting on a failure, it runs several programs over the same instance and reflects on the _contrast_ between the best and worst run of it — a comparison with the input held fixed. That contrast has to exist to be worth paying for, so SIMBA earns its extra rollouts on metrics that vary between runs and wastes them on ones that do not.

GEPA and OPRO update components separately. MIPRO instead searches combinations of per-component options, which lets it find options that work well only together. It screens configurations on minibatches and evaluates promising ones against the full validation set.

MIPRO's default multivariate TPE models complete configurations. Set `multivariate: false` to model each component independently; that usually needs fewer observations but cannot model interactions between components.

`BootstrapSearchOptimizer` writes no text at all. It harvests demonstrations from rollouts the metric already rewarded and searches over which set of them to keep, so it needs no proposal model and no textual feedback. Try it first when the instruction is roughly right and the output format is not.

`RandomSearchOptimizer` paraphrases components without using their scores. Run it as a baseline to check whether reflection improves enough to justify its model calls.

```ts
import { SimbaOptimizer } from "textopt/simba";
import { OproOptimizer } from "textopt/opro";
import { MiproOptimizer } from "textopt/mipro";
import { BootstrapSearchOptimizer } from "textopt/bootstrap-search";
import { RandomSearchOptimizer } from "textopt/random-search";
```

### What the benchmark says

`pnpm bench` runs every text-proposing optimizer over three offline tasks and twenty seeds, and writes [`bench/results/latest.json`](bench/results/latest.json). Scores are held-out; `p` is a paired sign-flip test against that task's winner.

| Task          | Winner  | Score | Runner-up      | Score | p     |
| ------------- | ------- | ----- | -------------- | ----- | ----- |
| `clean`       | `gepa`  | 0.729 | `simba`        | 0.381 | 0.000 |
| `noisy`       | `simba` | 0.511 | `gepa`         | 0.389 | 0.002 |
| `interacting` | `gepa`  | 0.750 | `randomSearch` | 0.235 | 0.000 |

The tasks differ only in their metric: `noisy` adds per-instance jitter to the same scoring function `clean` uses, and `interacting` pays only when two components are correct together. The split is the one SIMBA's premise predicts — mining disagreement costs rollouts that a noiseless metric never repays, and pays for itself once the metric is noisy.

Read these as evidence about the search, not about your task. The proposal model is a deterministic stand-in, so the benchmark holds proposal quality fixed and measures what each search does with it. Use `compare()` on your own task and metric before choosing.

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

`validationSet` defaults to `trainingSet`. That is the right default for a first run and the wrong number to report: the search selected against those instances, so `bestScore` is fitted to them. [Held-out evaluation](#held-out-evaluation) is how to find out what that cost.

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

`BootstrapSearchOptimizer` answers the cheapest question worth asking first — whether the instruction is already fine and the output format is what is failing — and it calls no proposal model to do it, so a run costs rollouts and nothing else. Reach for a reflective search once that has been ruled out, and pick between them with [`compare()`](#comparing-optimizers) under one budget rather than from the benchmark table above.

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

| Package                    | Contents                                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `textopt`                  | Shared contracts, evaluator, judge, `compare()`, demo utilities, cache, sampler, and RNG types. No runtime dependencies. |
| `textopt/gepa`             | `GepaOptimizer`, GEPA types, the pipeline adapter, and configurable selection, acceptance, and evaluation strategies.    |
| `textopt/simba`            | `SimbaOptimizer`, its advice prompt, and its bucket-ranking helpers.                                                     |
| `textopt/opro`             | `OproOptimizer` and score-history prompting.                                                                             |
| `textopt/mipro`            | `MiproOptimizer`, per-component option menus, TPE search, and minibatch screening.                                       |
| `textopt/bootstrap-search` | `BootstrapSearchOptimizer`, few-shot search that calls no proposal model.                                                |
| `textopt/random-search`    | `RandomSearchOptimizer`, a score-independent paraphrasing baseline.                                                      |
| `textopt/file-cache`       | An append-only evaluation cache that outlives the process. Uses `node:fs`.                                               |
| `textopt/testing`          | Deterministic fixtures for testing optimizers and adapters without an LLM.                                               |
| `@textopt/ai-sdk`          | Vercel AI SDK adapter for `generateText` and `generateObject`, including multi-step tool traces. **Beta, unpublished.**  |
| `@textopt/langchain`       | Adapter for LangChain runnables, chains, agents, and LangGraph graphs.                                                   |
| `@textopt/braintrust`      | autoevals scorer integration and a Braintrust logging decorator. **Beta, unpublished.**                                  |

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

This behavior is opt-in. The default is `buildReflectionPrompt`, adapted from the GEPA paper's reflection prompt rather than copied from it: it carries the same evidence and asks for the same thing, but tags and wording differ from the reference implementation's template, so proposals drawn from identical evidence will not match it.

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

`proposals.perIteration` is the setting that moves the bill: each proposal is priced at two minibatch evaluations, its parent's and its own, so raising it from one to three triples what an iteration costs before any child is swept ([Sizing a run](#sizing-a-run)). What that buys is width — every slot draws its own parent and its own minibatch, so the slots diagnose different failures and `concurrency` can run them at once, and `selection: "best"` then promotes only the strongest improving child of the batch. Pair it with `reflection.strategies`, which is what keeps two slots that landed on the same parent from writing the same revision.

`textopt/gepa` exports the following strategies: `paretoSelector`, `currentBestSelector`, `epsilonGreedySelector`, `topKParetoSelector`, `roundRobinComponentSelector`, `allComponentsSelector`, `improvementAcceptance`, `pairedPermutationAcceptance`, `fullEvaluationPolicy`, `subsampledEvaluationPolicy`, and `lowerBoundEvaluationPolicy`.

### Noisy metrics

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

Both are strictly more conservative, and that costs something real. On the benchmark's noiseless tasks the pair drops GEPA from 0.729 to 0.175, because every genuine improvement now has to clear a significance bar as well; on the noisy task it neither helps nor hurts, scoring 0.389 against plain GEPA's 0.389. Turn them on when you have measured the metric's own variance and found it large, not on principle.

A minibatch also has to be wide enough for the test to say anything: a sign-flip test over three instances cannot produce a p-value below 0.125, so at the default `minibatchSize` of 3 no proposal can ever be accepted at `alpha` below that.

## The other optimizers

Apart from SIMBA, these use the base `Adapter`: they need no reflective dataset. Their results include both `seedScore` and `bestScore`.

### SIMBA

```ts
import { SimbaOptimizer } from "textopt/simba";

const result = await new SimbaOptimizer({
  minibatchSize: 16,
  candidates: 4,
  maxSteps: 8,
  seed: 11,
}).optimize({
  seedCandidate,
  trainingSet,
  validationSet,
  adapter, // the base Adapter — no makeReflectiveDataset needed
  reflect,
  demoComponents: ["demos"], // optional; enables the appendDemo mutation
  maxMetricCalls: 900, // 250 reserved for finalists, 80 a step, over 50 validation instances
});

result.finalists; // the step winners, scored on the full validation set, best first
```

Each step samples several programs from the pool over one minibatch, ranks the instances by how much those programs disagreed, and mutates toward whichever run won. Two mutations are drawn at random per instance:

- **`appendDemo`** keeps the winning rollout as a few-shot example. Costs no model call. Requires `demoComponents`.
- **`appendRule`** shows the better and worse run of the same instance to `reflect` and appends the advice it returns to each instruction component.

Neither replaces text, so candidates accumulate; demonstrations are dropped at a Poisson rate so a growing block cannot crowd out the instruction. `strategies` pins the mutation to one of the two.

Ported from DSPy's SIMBA with two deliberate changes. A trajectory sample runs one program across the whole minibatch rather than resampling a program per instance, because the adapter owns decoding here and there is no temperature knob to vary — the variability comes from the program pool instead. And the percentile guards are strict rather than inclusive, so a step on which every rollout ties still produces a mutation instead of doing nothing at all.

Only the finalists are scored on the full validation set: the step winners are sampled evenly across the run, so early winners stay in the running. Those rollouts are reserved before the search starts, which is why a small `maxMetricCalls` buys fewer steps than the arithmetic suggests — see [Sizing a run](#sizing-a-run) for what the reserve costs.

Its batch defaults are wide on purpose: a step reads the disagreement between programs over a batch, and a narrow batch leaves little to rank. When a run has to get cheaper, lower `candidates` first — it shrinks both the step and the reserve, where `minibatchSize` shrinks the step alone and narrows the batch the ranking reads.

### Bootstrapped few-shot search

```ts
import { BootstrapSearchOptimizer } from "textopt/bootstrap-search";

const result = await new BootstrapSearchOptimizer({
  candidates: 16,
  maxDemos: 4,
  seed: 11,
}).optimize({
  seedCandidate: { instruction, demos: "" },
  trainingSet,
  validationSet,
  adapter,
  demoComponents: ["demos"],
  goldOutput: (datum) => datum.answer, // optional; enables the labels-only candidate
  maxMetricCalls: 1500, // 19 candidates swept over 50 validation instances, plus their harvests
});

result.candidates; // every set tried, with its source and how many demos it held
```

DSPy's `BootstrapFewShotWithRandomSearch`. It calls no model to write text: every candidate is assembled from outputs the system itself produced, so the search costs rollouts and nothing else. The fixed candidates come first — zero-shot, then labels-only when `goldOutput` is given, then one unshuffled full-size harvest — followed by shuffled harvests of random size.

Zero-shot stays in the running throughout. Demonstrations can hurt, and a search that cannot return "no demos" has no baseline to report against.

`candidates` is the whole search: each one is a fresh harvest and a full sweep, so it sets both the breadth and the bill. `demoMinScore` is the knob that surprises — a strict threshold does not cost less, it costs more, because a harvest keeps rolling out training instances until it has collected `maxDemos` of them or run out of set. If harvests come back with no demos, the seed cannot yet produce work its own metric rewards, and few-shot search is the wrong tool until it can.

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
  maxMetricCalls: 2000, // 4 proposals a round, each swept over 50 validation instances
});

result.trajectory; // every candidate scored, in the order it was tried
```

By default, every proposal is scored on the full `validationSet`. With `scoringSetSize`, proposals are screened on a fixed subset of `trainingSet`, and the incumbent receives a full sweep every `fullEvalInterval` rounds. In a 30-instance validation set, screening on 12 instances halved rollout count without reducing the measured best score.

The meta-prompt lists the strongest attempts in ascending score order, placing the best attempt nearest the request. `scoreScale` converts scores to integers (100 by default), because models distinguish values such as 41 and 68 more reliably than 0.41 and 0.68.

Rounds are where OPRO gets its signal: every proposal in a round sees the same history, so `proposalsPerRound` widens a round rather than deepening the search, and the history a later prompt reads only grows between rounds. A run also moves one component per round, in turn, so a two-component candidate needs twice the rounds to revise each as often. Budget for rounds first, then set `proposalsPerRound` to what a round can afford — `maxReflectionCalls` caps the two together, at `maxReflectionCalls / proposalsPerRound` rounds.

### MIPRO

```ts
const result = await new MiproOptimizer({
  instructionsPerComponent: 5,
  maxTrials: 30,
  seed: 11,
}).optimize({
  seedCandidate,
  trainingSet,
  validationSet,
  adapter,
  reflect,
  demoComponents: ["demos"], // menu bootstrapped from the training set
  maxMetricCalls: 1600, // 30 trials of 35, six sweeps of 50, and the demo harvests
});

result.menu; // the space that was searched, per component
result.observations; // every configuration tried, and which earned a full sweep
```

MIPRO first builds a menu for each component from the seed text and variants generated by `reflect`. A TPE surrogate then proposes configurations from those menus. Trials run on minibatches; selected configurations receive a full validation sweep.

`componentOptions` adds menu entries without reflection calls. `demoComponents` builds menus of few-shot blocks from successful training rollouts, allowing instructions and demonstrations to be searched together.

Every `fullEvalInterval` trials, MIPRO fully evaluates the unswept configuration with the highest average minibatch score. Averaging repeated observations reduces the effect of a lucky minibatch.

The space is the product of the menus, so it grows multiplicatively with components while `maxTrials` grows by hand — three components of six options each is 216 configurations, and the default thirty trials sees a seventh of them. The surrogate also spends its first ten trials sampling at random before it models anything, so a short run is mostly random search with extra steps. Give it trials in proportion to the menu, or trim the menu with `instructionsPerComponent`.

### Random search

```ts
const result = await new RandomSearchOptimizer({
  variants: 4,
}).optimize({
  seedCandidate,
  trainingSet,
  validationSet,
  adapter,
  reflect,
  maxMetricCalls: 1000, // 4 variants a round, each swept over 50 validation instances
});
```

Random search paraphrases one component per round and keeps the highest-scoring candidate. Its prompt receives no performance data. Compare it with a reflective optimizer under the same metric budget to measure the benefit of reflection.

Being a baseline is the whole configuration: give it the `maxMetricCalls` and `validationSet` of the run it stands against, and change nothing else. A baseline on a smaller budget answers a different question than the one being asked of it. [`compare()`](#comparing-optimizers) runs both over the same seeds and ranks them on the held-out score, but each entrant builds its own task, so keeping the budgets equal is still yours to do.

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

## Comparing optimizers

A difference in means over a handful of seeds is usually noise. `compare()` runs each entrant over the same seeds, ranks them on `testScore` where a run reports one, and reports a paired sign-flip p-value against the winner:

```ts
import { compare } from "textopt";

const comparison = await compare({
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  concurrency: 4,
  entrants: {
    gepa: ({ seed }) => new GepaOptimizer({ seed }).optimize(task()),
    opro: ({ seed }) => new OproOptimizer({ seed }).optimize(task()),
  },
});

comparison.winner; // highest mean score
comparison.summaries; // mean, sd, min, max, rollouts, cost, pValueVsWinner
comparison.runs; // every individual run
```

Entrants are functions of a seed, not optimizer instances: the seed is constructor config, and every optimizer here is deterministic given one, so comparing two entrants at a single seed compares two anecdotes. Build a fresh task inside each entrant — a shared reflection model with internal state would make each result depend on the runs before it.

Ranking on `testScore` matters. The validation score is the number the search selected against for its whole run, so an entrant that overfits looks strongest on exactly the number it fitted.

## Judging with a model

When the metric cannot be written as a string match, `createJudge` builds one from a model and returns written feedback alongside the score, which is what reflective search actually runs on:

```ts
import { createJudge } from "textopt";

const judge = createJudge<Ticket, string>({
  model: reflect,
  scale: 5,
  criteria: [
    {
      name: "accuracy",
      description: "Every claim is supported by the ticket.",
    },
    { name: "tone", description: "Direct and free of filler." },
  ],
});

const { score, feedback, objectiveScores } = await judge({
  input: ticket,
  output: answer,
});
```

Each criterion is graded on a small integer scale and normalized afterwards, because models discriminate between 2 and 4 far more reliably than between 0.4 and 0.8. Per-criterion scores are returned as `objectiveScores`, so a Pareto frontier can be taken over them; the aggregate `score` is their mean.

The prompt asks for feedback addressed to the _instructions_ rather than to the graded output. "The instruction never says to state the refund window" is something a rewriting model can act on; "this answer should have mentioned the refund window" is not. A criterion the judge failed to grade comes back as a transient score, so the instance is retried rather than recorded as a zero.

## Multi-module pipelines

When a system runs several modules in sequence and each has its own instruction, reflection is only as good as the evidence it sees — and the evidence a module needs is what _it_ received and produced, not the pipeline's input and final answer. `createPipelineAdapter` builds that attribution:

```ts
import { createPipelineAdapter } from "textopt/gepa";

const adapter = createPipelineAdapter<Ticket, string, "planner" | "writer">({
  modules: [
    {
      component: "planner",
      run: ({ instruction, datum }) => plan(instruction, datum),
    },
    {
      component: "writer",
      run: ({ instruction, input }) => write(instruction, input),
    },
  ],
  score: ({ datum, output, steps }) => judgeAnswer(datum, output, steps),
  concurrency: 4,
});
```

Each module's output is threaded into the next, and the reflective dataset gives each component only its own step. The feedback is end-to-end and every module sees the same string: a metric scores the final output, so nothing in a score alone says which module lost the point. `score` is handed the whole trace for callers who can attribute better.

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
```

- **`maxCostUsd`** exists because reflective search grows the text it optimizes, so late rollouts cost more than early ones. It is checked between evaluations and reads whatever usage the adapter reported; an adapter that reports none can never trigger it. `priceUsage` fills in `costUsd` on a rollout's usage from a `TokenPricing` table; adapters call it so there is something to read.
- **`maxWallClockMs`** exists because a run behind a rate limit spends almost nothing and takes as long as the provider makes it take. This is what makes an optimizer safe to put behind a request timeout or a nightly job. `stopReason` is `"deadlineReached"`.
- **`retry`** re-runs instances the adapter marked `transient`. A rate limit or a 5xx otherwise costs the instance either an unexplained zero or a hole in the candidate's coverage. Retries are charged like any other rollout but never overdraw the budget. Defaults to two attempts, 500 ms apart, doubling.

## Caching, checkpoints, resume

- **Caching.** Cache keys include the split, complete candidate, and instance ID. Cache hits do not count against the metric budget. Instance IDs default to a content hash, with the row position as a fallback for values that cannot be serialized. Provide `instanceId` for non-serializable data or readable trace IDs. Set `cache: false` to disable caching.
- **`cacheNamespace`.** A cached score is a measurement of a whole system, not of a candidate. Set `cacheNamespace` to name the model id, decoding settings, and scorer version, and change it whenever anything outside the candidate text changes — otherwise a run silently reuses scores measured under a system it is no longer running.
- **Durable caching.** `createFileCache({ path })` from `textopt/file-cache` is an append-only log that outlives the process, so a crashed run, a re-run with a changed budget, and a second experiment over the same validation set do not pay for identical rollouts again. It needs `node:fs`; for Redis or SQLite, implement `EvaluationCache` yourself.
- **Checkpoints.** Every optimizer takes `onCheckpoint` and `resumeFrom`, and returns its final `snapshot`. Each restores exactly the state that is expensive or unrepeatable: GEPA's candidate pool, rejections and merge state; MIPRO's option menus and surrogate observations; OPRO's screening slice and score histories; SIMBA's program pool and step winners; the search's budget, RNG, sampler position and cached scores throughout. A fingerprint refuses a checkpoint from a different seed candidate, instance set, or random seed. Snapshots are plain JSON and are never mutated by the run that resumes from them.
- **Events.** `onEvent` receives a discriminated union per optimizer. Every one of them emits `start`, `evaluation`, and `finish`.
- **Result.** Every result carries `bestCandidate`, `bestScore`, `metricCalls`, `usage`, `stopReason`, and `snapshot`, plus whatever its own search can report.

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
| `pnpm --filter textopt-examples simba`      | none                | Mini-batch ascent against a metric that returns a different number every time.                               |
| `pnpm --filter textopt-examples bootstrap`  | none                | Few-shot search with no proposal model, and why more demos is not better.                                    |
| `pnpm --filter textopt-examples compare`    | none                | Three optimizers over the same seeds, with p-values.                                                         |
| `pnpm --filter textopt-examples judge`      | `ANTHROPIC_API_KEY` | A model-graded metric built from named criteria.                                                             |
| `pnpm --filter textopt-examples pipeline`   | `ANTHROPIC_API_KEY` | Two modules in sequence, each with its own instruction.                                                      |

Keys are read from a root `.env` if present.

## Development

Requires Node >= 20 and pnpm 10.

```bash
pnpm install
pnpm test          # vitest, runs against source via workspace aliases
pnpm build         # tsdown, per package
pnpm typecheck     # build, then tsc --noEmit everywhere
pnpm format        # prettier --write
pnpm lint:packages # publint and are-the-types-wrong, against the built output
pnpm bench         # 20-seed sweep, rewrites bench/results/latest.json
```

`pnpm bench` takes a few minutes and makes no network calls. Its output is committed, so a change that moves an optimizer's numbers shows up in the diff.

## License

MIT
