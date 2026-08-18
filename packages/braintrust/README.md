# @textopt/braintrust

Braintrust scorers and experiment logging for [textopt](../../README.md).

Two independent pieces, usable together or apart:

- `createBraintrustScorer` turns autoevals (or braintrust) scorers into the optimizer's metric.
- `withBraintrustLogging` decorates any adapter so every rollout lands in Braintrust.

`autoevals` and `braintrust` are yours to install; this package only matches their shapes structurally.

## Scoring

```ts
import { createBraintrustScorer } from "@textopt/braintrust";
import { ExactMatch, Levenshtein } from "autoevals";

const score = createBraintrustScorer<string>({
  scorers: [ExactMatch, Levenshtein],
  weights: { ExactMatch: 3 },
});

const result = await score({ output, expected });
// { score, feedback, objectiveScores: { ExactMatch, Levenshtein } }
```

The returned function produces a `ScoreResult`, so an adapter's `score` can await it directly. Map your own row to its arguments at the call site: the scorer wants a non-null `output` and the `expected` value to compare against.

The number is the weighted mean, but the more useful output is the rest. Each scorer's metadata (an LLM judge's rationale, a diff, a validation error) is carried through as `feedback`, which is what the reflection model actually reads, and each scorer's own number is carried through as `objectiveScores`, which is what a per-objective frontier is built from.

| Option          | Default                          | Effect                                                                 |
| --------------- | -------------------------------- | ---------------------------------------------------------------------- |
| `scorers`       | required                         | The scorers to run. Each may return a `Score` object or a bare number. |
| `weights`       | `1` per scorer                   | Relative weight by scorer name. Must be finite and non-negative.       |
| `buildFeedback` | one line per scorer              | Overrides how scorer output becomes reflection feedback.               |
| `isTransient`   | every failure is the candidate's | Classifies a thrown scorer error as infrastructure.                    |

Two scorers reporting the same name throws, since the blended score would count both while `objectiveScores` kept only the last. When a scorer fails and `isTransient` says the cause was infrastructure, the composite is computed from whichever scorers survived and the whole result is marked `transient`, so it is never cached.

## Logging

```ts
import { withBraintrustLogging } from "@textopt/braintrust";
import { initLogger } from "braintrust";

const adapter = withBraintrustLogging({
  adapter: baseAdapter,
  logger: initLogger({ projectName: "ticket-routing" }),
  toExpected: (datum) => datum.label,
});
```

Because it decorates the adapter rather than replacing it, this composes with [`@textopt/ai-sdk`](../ai-sdk), [`@textopt/langchain`](../langchain), or an adapter you wrote yourself. Extra methods survive the wrapping, so a `GepaAdapter` stays a `GepaAdapter`, and the component names stay narrow instead of widening back to `string`.

Every logged event carries the candidate, the instance feedback, and the `iteration`, `phase`, `split`, and `candidateId` it came from, so an experiment can be grouped by where in the run each rollout sits.

| Option       | Default        | Effect                                                                                |
| ------------ | -------------- | ------------------------------------------------------------------------------------- |
| `adapter`    | required       | The adapter to wrap.                                                                  |
| `logger`     | required       | Anything with `log(event)`. Both a braintrust `Experiment` and a `Logger` satisfy it. |
| `metadata`   | none           | Merged into every event's metadata.                                                   |
| `toInput`    | the row itself | Maps a dataset row to the logged input.                                               |
| `toExpected` | omitted        | Maps a dataset row to the logged expected value.                                      |

Logging never fails a rollout. An unreachable logger degrades to a warning.

Types: `BraintrustScorerOptions`, `BraintrustScorerFn`, `BraintrustScorerArgs`, `BraintrustScoreLike`, `BraintrustLoggingOptions`, `BraintrustLoggerLike`, `BraintrustEvent`.

## License

MIT
