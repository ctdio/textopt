# @textopt/langchain

LangChain adapter for [textopt](https://github.com/ctdio/textopt#readme).

The adapter rebuilds a LangChain runnable, chain, agent, or LangGraph graph for each candidate. Candidate components can supply prompts, tool descriptions, routing instructions, or other text read by the runnable.

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
  trainingSet,
  adapter,
  reflect,
  maxMetricCalls: 150,
});
```

`score` returns a `ScoreResult`. GEPA uses its optional `feedback` field during reflection, so include a description of the failure when possible.

## Options

| Option              | Default                   | Effect                                                                                     |
| ------------------- | ------------------------- | ------------------------------------------------------------------------------------------ |
| `buildRunnable`     | required                  | Rebuilds the chain with the candidate's text injected into its prompts.                    |
| `score`             | required                  | Scores one rollout, given the `datum`, `output`, and `trace`.                              |
| `toInput`           | the row itself            | Maps a dataset row to the chain's input.                                                   |
| `concurrency`       | `8`                       | In-flight rollouts.                                                                        |
| `includeChainSteps` | `false`                   | Includes per-runnable chain spans in traces.                                               |
| `componentRunNames` | none                      | Maps component names to run names and filters each component's evidence to matching steps. |
| `isTransient`       | `false` for all errors    | Identifies infrastructure errors whose fallback scores must not be cached.                 |
| `buildRecord`       | default reflective record | Builds a custom reflective record.                                                         |

## Tracing

Each rollout includes `textopt_iteration`, `textopt_phase`, `textopt_split`, and `textopt_candidate_id` in its LangChain metadata. With a tracer configured, these fields can filter and group runs in LangSmith.

Traces include LLM, tool, and retriever spans with run IDs, parent run IDs, inputs, outputs, and errors. Set `includeChainSteps` to include chain spans.

Tracing callbacks are attached only when the optimizer requests traces, typically for reflection minibatches. During validation sweeps, `score` receives a trace with no steps and should score the output directly.

Types: `LangChainAdapterOptions`, `LangChainTrace`, `LangChainTraceStep`, `LangChainStepType`, `LangChainEvidence`, `LangChainScore`.

## License

MIT
