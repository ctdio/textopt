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

## Install

```bash
npm install textopt
```

`textopt` and `@textopt/langchain` are the packages this repository releases. The test suite has more than 550 tests and runs without network access.

`@textopt/ai-sdk` and `@textopt/braintrust` are beta and are not published to npm. Use them from a checkout of this repository until their interfaces settle.

## Concepts

**A candidate is a record of named strings.** Components can be system prompts, tool descriptions, routing rules, few-shot blocks, regexes, or configuration. Component names are inferred from the seed candidate, so a misspelled name fails type checking rather than silently optimizing nothing.

**An adapter runs and scores the system.** It returns one score per instance, and may return textual feedback alongside it. That feedback is what reflective search reads — a bare `0.0` tells a rewriting model nothing about what to change. See [Adapters and metrics](docs/adapters.md) for the interface and the AI SDK, LangChain, and Braintrust adapters.

**The reflection model is any text-in, text-out function.** `reflect` implements `({ prompt, signal }) => Promise<string>`, so a vendor SDK, a LangChain chat model, a local model, or a deterministic stub all satisfy it. The model under optimization is usually cheaper than the one revising it.

**`maxMetricCalls` is the budget.** It bounds scored rollouts; cache hits do not count, and reflection calls have their own limit. A run that cannot afford its next unit of work stops and reports `stopReason: "budgetExhausted"` with whatever it found — it does not throw. [Tuning a run](docs/tuning.md) is the arithmetic for pricing one before starting it.

**Three sets, one of them held out.** The search proposes against `trainingSet` and selects against `validationSet`, which defaults to it. `bestScore` is therefore fitted to the instances that chose the winner; pass a `testSet` to get a number that no candidate was selected against.

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

## Choosing an optimizer

All six use the same `Optimizer` interface, adapter, and budget accounting, so switching one for another is a change of import and constructor.

| Optimizer                  | Required signal                   | Use it for                                                        |
| -------------------------- | --------------------------------- | ----------------------------------------------------------------- |
| `GepaOptimizer`            | per-instance **textual feedback** | revising text from written explanations of failures               |
| `SimbaOptimizer`           | per-instance **textual feedback** | noisy metrics, where the same instance scores differently per run |
| `OproOptimizer`            | a **scalar** score                | proposing text from a history of scored attempts                  |
| `MiproOptimizer`           | a **scalar** score                | searching combinations of interacting component options           |
| `BootstrapSearchOptimizer` | a **scalar** score                | few-shot demonstrations, with no proposal model at all            |
| `RandomSearchOptimizer`    | a **scalar** score                | establishing a score-independent paraphrasing baseline            |

`BootstrapSearchOptimizer` answers the cheapest question first — whether the instruction is already fine and consistency is what is failing — and calls no proposal model to do it. Reach for a reflective search once that is ruled out, and pick between them with [`compare()`](docs/evaluation.md#comparing-optimizers) rather than from a table.

[Optimizers](docs/optimizers.md) covers what each one reads and how to configure it.

## Packages

| Package               | Contents                                                                                                               |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `textopt`             | Every optimizer, plus the shared contracts, evaluator, judge, `compare()`, demo utilities, and cache. No dependencies. |
| `@textopt/langchain`  | Adapter for LangChain runnables, chains, agents, and LangGraph graphs.                                                 |
| `@textopt/ai-sdk`     | Vercel AI SDK adapter for `generateText` and `generateObject`. **Beta, unpublished.**                                  |
| `@textopt/braintrust` | autoevals scorer integration and a Braintrust logging decorator. **Beta, unpublished.**                                |

Each optimizer ships behind its own subpath — `textopt/gepa`, `textopt/simba`, `textopt/opro`, `textopt/mipro`, `textopt/bootstrap-search`, `textopt/random-search` — alongside `textopt/file-cache` for a durable cache and `textopt/testing` for fixtures that need no LLM. [`packages/core/README.md`](packages/core/README.md) is the API reference.

## Documentation

| Page                                     | Covers                                                                                   |
| ---------------------------------------- | ---------------------------------------------------------------------------------------- |
| [Optimizers](docs/optimizers.md)         | Choosing one, how GEPA works, configuring each search, few-shot demos                    |
| [Adapters and metrics](docs/adapters.md) | The adapter interface, framework adapters, the reflection model, model judges, pipelines |
| [Tuning a run](docs/tuning.md)           | What a run costs, noisy metrics, cost and wall-clock budgets, caching and resume         |
| [Measuring a result](docs/evaluation.md) | Held-out evaluation and `compare()`                                                      |
| [Benchmark](docs/benchmark.md)           | What every optimizer scores on four offline tasks over twenty seeds                      |
| [Examples](examples/README.md)           | Runnable scripts, offline and against real providers                                     |

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

[AGENTS.md](AGENTS.md) is the guide for working in this repository.

## License

MIT
