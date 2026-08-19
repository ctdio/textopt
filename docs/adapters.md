# Adapters and metrics

An adapter is how an optimizer runs and scores your system. Everything that
feeds a search — the framework it calls, the model that revises text, a
model-graded metric, per-module attribution — hangs off it.

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
