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

Most systems do not need this written by hand. [`createPromptAdapter`](#one-prompt) covers one prompt and one component, [`createPipelineAdapter`](#multi-module-pipelines) covers several modules in sequence, and the framework adapters below cover the AI SDK and LangChain. Implement the interface directly when none of them describes your system.

`evaluate` returns one score per instance and may include textual feedback. GEPA uses that feedback during reflection.

- **`args.run`** identifies the rollout's `iteration`, `phase`, `split`, and `candidateId`. Forward it to your tracing system.
- **`args.onRollout`** is what turns a batch into progress. Call it as each rollout settles and the optimizer emits a `rollout` event per call; skip it and the run reports nothing between the start of a validation sweep and its end, which against a slow provider is minutes of silence indistinguishable from a hung process. The adapters below already call it, and so does anything built on `mapWithConcurrency` — pass it as `onSettled`.
- **`transient`** marks scores caused by infrastructure failures such as rate limits or 5xx responses. Transient scores are not cached.

```ts
evaluate: async ({ batch, candidate, onRollout, signal }) => {
  const scored = await mapWithConcurrency({
    items: batch,
    limit: 4,
    signal,
    onSettled: onRollout,
    task: (datum) => runAndScore(candidate, datum),
  });
  // …
};
```

### Vercel AI SDK

Beta, and not published to npm: use it from a checkout of this repository until its interface settles.

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

Beta, and not published to npm: use it from a checkout of this repository until its interface settles.

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

## One prompt

Most systems under optimization are a single call with a single instruction, and `createPromptAdapter` is that case named:

```ts
import { createPromptAdapter } from "textopt/gepa";

const adapter = createPromptAdapter<Ticket, string>({
  input: (datum) => datum.text,
  run: ({ instruction, input, signal }) => classify(instruction, input, signal),
  score: ({ datum, output }) => gradeLabel(datum, output),
  concurrency: 4,
});
```

`run` receives the candidate's text as `instruction`, and `input(datum)` is what the prompt receives — the datum itself by default. That default is worth overriding whenever a row carries fields the system never sees: reflection reads the record, and a record carrying the label teaches the reflection model to write rules about an answer key the task model was never given.

The candidate must have exactly one component, and the adapter reads which one off the candidate rather than being told. That is the check, not a limitation: a second component nobody runs is text the search rewrites every iteration for no effect, a budget spent on proposals that cannot move the score and a run that reports nothing unusual. Two instructions want the section below.

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

`createPromptAdapter` above is this helper with a single module, and the two share every behaviour described here.

### Text a component can countermand

A component is mutable text, and everything around it in the prompt is not. That asymmetry is easy to forget, and it costs a run quietly: an output contract written into the frozen text — "separate messages with `---`" — can be overridden by the very instruction the search is rewriting, because reflection is asked to fix failures and knows nothing about which of the words it is looking at are yours. A run that does this drifts to a candidate carrying `do not use dividers like ---`, the contract stops being met, and the component of the metric that scored it degenerates to a constant while the aggregate keeps improving.

Two things prevent it, and they are worth doing together:

- **Put the contract where the metric enforces it.** A format requirement that is scored is a requirement the search cannot drift away from — it costs points immediately. One that is only stated in the prompt is a suggestion the search is free to overrule. See [Building a metric](./metric-preflight.md).
- **Watch the objective, not the aggregate.** Score the contract as its own entry in `objectiveScores` and read it off the `candidateAccepted` event. A channel collapsing to a constant while the total climbs is exactly what this failure looks like from the outside, and it is unreadable in the aggregate alone.
