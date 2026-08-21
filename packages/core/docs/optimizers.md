# Optimizers

What each search reads, and what its knobs do. See [Tuning a run](./tuning.md)
for what a run costs and [Benchmark](./benchmark.md) for how the six compare on
offline tasks.

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

[Benchmark](./benchmark.md) reports what each search does on four offline tasks; read it as evidence about the search rather than about your task, and run [`compare()`](./evaluation.md#comparing-optimizers) on your own metric before choosing.

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

`optimize` takes a `GepaTask`. Required fields are `seedCandidate`, `trainingSet`, `adapter`, `reflect`, and `maxMetricCalls`. `validationSet` defaults to `trainingSet`; `testSet` is optional and held out. Other options are `componentSelector`, `batchSampler`, `valEvaluationPolicy`, `instanceId`, `cache`, `reporters`, `onCheckpoint`, `resumeFrom`, and `signal`.

Component names are inferred from `seedCandidate`. Other positions use `NoInfer`, so misspelled component names fail type checking.

`proposals.perIteration` is the setting that moves the bill: each proposal is priced at two minibatch evaluations, its parent's and its own, so raising it from one to three triples what an iteration costs before any child is swept ([Sizing a run](./tuning.md#sizing-a-run)). What that buys is width — every slot draws its own parent and its own minibatch, so the slots diagnose different failures and `concurrency` can run them at once, and `selection: "best"` then promotes only the strongest improving child of the batch. Pair it with `reflection.strategies`, which is what keeps two slots that landed on the same parent from writing the same revision.

`textopt/gepa` exports the following strategies: `paretoSelector`, `currentBestSelector`, `epsilonGreedySelector`, `topKParetoSelector`, `roundRobinComponentSelector`, `allComponentsSelector`, `improvementAcceptance`, `pairedPermutationAcceptance`, `fullEvaluationPolicy`, `subsampledEvaluationPolicy`, and `lowerBoundEvaluationPolicy`.

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

Before optimization, `harvestFewShotExamples` evaluates the seed candidate on `trainingSet` and keeps high-scoring rollouts:

```ts
import { harvestFewShotExamples } from "textopt";

const { block, demos, metricCalls } = await harvestFewShotExamples({
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

Only the finalists are scored on the full validation set: the step winners are sampled evenly across the run, so early winners stay in the running. Those rollouts are reserved before the search starts, which is why a small `maxMetricCalls` buys fewer steps than the arithmetic suggests — see [Sizing a run](./tuning.md#sizing-a-run) for what the reserve costs.

Its batch defaults are wide on purpose: a step reads the disagreement between programs over a batch, and a narrow batch leaves little to rank. When a run has to get cheaper, lower `candidates` first — it shrinks both the step and the reserve, where `minibatchSize` shrinks the step alone and narrows the batch the ranking reads.

`concurrency` overlaps the two places a step's work is independent: scoring the candidates it built, and the finalist sweeps at the end. The trajectory samples and the mutations that read them stay in sequence — each reads what the one before it wrote — so the ceiling on what it buys is the reserve and the scoring, not the whole step. See [Concurrency](./tuning.md#concurrency).

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

`concurrency` overlaps a candidate's sweep with the harvest of the candidates behind it, which is most of the wall clock on a run whose harvests are as expensive as its sweeps. Harvesting itself stays in plan order, because every harvest draws from the same random stream. It is ignored when `stopAtScore` is set, and it moves checkpointing from once per candidate to once per wave — see [Concurrency](./tuning.md#concurrency).

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

Because every proposal in a round sees the same history and is screened against the same incumbent, `concurrency` runs a whole round — proposals and screens together — at once, and the round still records its attempts in the order it drew them.

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

`concurrency` runs a round's variants at once, proposal and sweep together, and is the one setting here that changes nothing about the search — the round still accepts a variant only if it beat every variant drawn before it.

Being a baseline is the whole configuration: give it the `maxMetricCalls` and `validationSet` of the run it stands against, and change nothing else. A baseline on a smaller budget answers a different question than the one being asked of it. [`compare()`](./evaluation.md#comparing-optimizers) runs both over the same seeds and ranks them on the held-out score, but each entrant builds its own task, so keeping the budgets equal is still yours to do.
