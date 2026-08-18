# @textopt/langchain

LangChain adapter for [textopt](../../README.md).

Runs a LangChain runnable, chain, agent, or LangGraph graph as the system under optimization. The candidate is injected by rebuilding the runnable, so anything downstream that reads candidate text is optimizable: prompts, tool descriptions, routing instructions.

`@langchain/core` is a peer dependency (`>=0.3.0 <2`).

## Usage

```ts
import { GepaOptimizer } from "textopt/gepa";
import { createLangChainAdapter } from "@textopt/langchain";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

const adapter = createLangChainAdapter<Ticket, string>({
  buildRunnable: (candidate) =>
    ChatPromptTemplate.fromMessages([
      ["system", candidate.system ?? ""],
      ["human", "{text}"],
    ])
      .pipe(model)
      .pipe(new StringOutputParser()),

  toInput: (datum) => ({ text: datum.text }),

  score: ({ datum, output }) =>
    output === datum.label
      ? { score: 1, feedback: `Correct: ${datum.label}.` }
      : {
          score: 0,
          feedback: `Predicted "${output}", expected "${datum.label}".`,
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

`score` returns a `ScoreResult`, and its `feedback` is the load-bearing field. Say what went wrong, not just how wrong it was.

## Options

| Option              | Default                          | Effect                                                                                                                                             |
| ------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `buildRunnable`     | required                         | Rebuilds the chain with the candidate's text injected into its prompts.                                                                            |
| `score`             | required                         | Scores one rollout, given the `datum`, `output`, and `trace`.                                                                                      |
| `toInput`           | the row itself                   | Maps a dataset row to the chain's input.                                                                                                           |
| `concurrency`       | `8`                              | In-flight rollouts.                                                                                                                                |
| `includeChainSteps` | `false`                          | Adds per-runnable chain spans to the trace. Noisy.                                                                                                 |
| `componentRunNames` | none                             | Component name to LangChain run name. That component's reflective records then show only the steps with that run name, instead of the whole trace. |
| `isTransient`       | every failure is the candidate's | Classifies a thrown error as infrastructure, so its zero is not cached.                                                                            |
| `buildRecord`       | a record with trace evidence     | Replaces the reflective record wholesale.                                                                                                          |

## Tracing

Every rollout carries `textopt_iteration`, `textopt_phase`, `textopt_split`, and `textopt_candidate_id` in its LangChain metadata. This is inert without a tracer configured, and it is the difference between a LangSmith project full of anonymous rollouts and one you can filter down to the iteration whose score moved.

Traces collect LLM, tool, and retriever spans by default, each carrying its `runId`, `parentRunId`, inputs, outputs, and error. Chain spans join them behind `includeChainSteps`.

The callbacks are only attached when the optimizer asks for traces, which it does on the minibatch it reflects over. On a validation sweep `score` receives a trace with no steps, so score on the output there rather than on the trace.

Types: `LangChainAdapterOptions`, `LangChainTrace`, `LangChainTraceStep`, `LangChainStepType`, `LangChainEvidence`, `LangChainScore`.

## License

MIT
