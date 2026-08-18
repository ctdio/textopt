# @textopt/ai-sdk

Vercel AI SDK adapter for [textopt](../../README.md).

Optimizes the text components of an AI SDK call (system prompts, tool descriptions, output instructions) by re-running your `generateText` or `generateObject` call with each candidate. Multi-step agent runs keep their tool calls and results in the trace, which is what the reflection model reads when it diagnoses a failure.

This package does not depend on `ai` at runtime, and does not declare it as a peer. The SDK result types are matched structurally, so the adapter tolerates version drift.

## Usage

```ts
import { GepaOptimizer } from "textopt/gepa";
import { createAiSdkAdapter } from "@textopt/ai-sdk";
import { generateText } from "ai";

const adapter = createAiSdkAdapter<Ticket>({
  run: ({ candidate, datum, signal }) =>
    generateText({
      model,
      system: candidate.system ?? "",
      prompt: datum.text,
      abortSignal: signal,
    }),

  score: ({ datum, output }) =>
    output === datum.label
      ? { score: 1, feedback: `Correct: ${datum.label}.` }
      : {
          score: 0,
          feedback: `Predicted "${output}" but the correct queue is "${datum.label}".`,
        },
});

const result = await new GepaOptimizer().optimize({
  seedCandidate: { system: "Classify the support ticket." },
  trainset,
  adapter,
  reflect,
  maxMetricCalls: 150,
});
```

`score` returns a `ScoreResult`, and its `feedback` is the load-bearing field. A number tells the search how wrong a candidate was; the feedback tells the reflection model what was wrong.

## Options

| Option        | Default                                                   | Effect                                                                      |
| ------------- | --------------------------------------------------------- | --------------------------------------------------------------------------- |
| `run`         | required                                                  | Runs the system for one dataset row. Return the AI SDK result directly.     |
| `score`       | required                                                  | Scores one rollout, given the `datum`, `output`, raw `result`, and `trace`. |
| `toOutput`    | `result.text`, or `""` when absent                        | Extracts the output. Required when optimizing structured output.            |
| `concurrency` | `8`                                                       | In-flight rollouts.                                                         |
| `isTransient` | every failure is the candidate's                          | Classifies a thrown error as infrastructure, so its zero is not cached.     |
| `buildRecord` | a record, with step evidence on multi-step or failed runs | Replaces the reflective record wholesale, `evidence` included.              |

`run` also receives the `EvaluationContext` under `run`, holding `iteration`, `phase`, `split`, and `candidateId`. Forward it into whatever tracing the call already has, or the run becomes thousands of indistinguishable rollouts.

The `candidate` passed to `run` is keyed by `string`, since this factory never sees the seed candidate. A component read here is not checked against the ones the optimizer was given.

## Also exported

`summarizeRun(result)` flattens an AI SDK result into the compact `AiSdkTrace` the adapter builds internally: per-step text, finish reasons, tool calls, tool results, and usage. Useful when scoring wants the step detail rather than just the final text. It leaves `durationMs` at `0`, since only the adapter times the rollout.

Types: `AiSdkAdapterOptions`, `AiSdkTrace`, `AiSdkTraceStep`, `AiSdkEvidence`, `AiSdkResultLike`, `AiSdkStepLike`, `AiSdkUsageLike`, `AiSdkToolCallLike`, `AiSdkToolResultLike`.

## License

MIT
