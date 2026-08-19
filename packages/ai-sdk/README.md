# @textopt/ai-sdk

Vercel AI SDK adapter for [textopt](../../README.md).

The adapter evaluates candidate system prompts, tool descriptions, and output instructions by rerunning a `generateText` or `generateObject` call. Traces from multi-step runs include tool calls and results for use during reflection.

The package matches AI SDK result types structurally and does not depend on `ai` at runtime or declare it as a peer dependency.

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
  trainingSet,
  adapter,
  reflect,
  maxMetricCalls: 150,
});
```

`score` returns a `ScoreResult`. GEPA uses its optional `feedback` field to revise the candidate; a scalar score alone only indicates relative performance.

## Options

| Option        | Default                            | Effect                                                                      |
| ------------- | ---------------------------------- | --------------------------------------------------------------------------- |
| `run`         | required                           | Runs the system for one dataset row. Return the AI SDK result directly.     |
| `score`       | required                           | Scores one rollout, given the `datum`, `output`, raw `result`, and `trace`. |
| `toOutput`    | `result.text`, or `""` when absent | Extracts the output. Required when optimizing structured output.            |
| `concurrency` | `8`                                | In-flight rollouts.                                                         |
| `isTransient` | `false` for all errors             | Identifies infrastructure errors whose fallback scores must not be cached.  |
| `buildRecord` | default reflective record          | Builds a custom reflective record, including its `evidence` field.          |

`run` also receives an `EvaluationContext` containing `iteration`, `phase`, `split`, and `candidateId`. Forward these fields to your tracing system to identify each rollout.

The `candidate` passed to `run` is keyed by `string` because the adapter factory does not receive the seed candidate. Component names accessed here are therefore not checked against the optimizer's candidate type.

## Also exported

`summarizeRun(result)` converts an AI SDK result to `AiSdkTrace`, including per-step text, finish reasons, tool calls, tool results, and usage. Use it when scoring needs step data. `durationMs` is `0` because timing is only available when the adapter runs the rollout.

Types: `AiSdkAdapterOptions`, `AiSdkTrace`, `AiSdkTraceStep`, `AiSdkEvidence`, `AiSdkResultLike`, `AiSdkStepLike`, `AiSdkUsageLike`, `AiSdkToolCallLike`, `AiSdkToolResultLike`.

## License

MIT
