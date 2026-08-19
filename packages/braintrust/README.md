# @textopt/braintrust

Braintrust scorers and experiment logging for [textopt](https://github.com/ctdio/textopt#readme).

The package provides two independent integrations:

- `createBraintrustScorer` adapts autoevals or Braintrust scorers to textopt's metric interface.
- `withBraintrustLogging` decorates an adapter and logs its rollouts to Braintrust.

Install `autoevals` or `braintrust` separately. This package matches their interfaces structurally.

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

The returned function produces a `ScoreResult` and can be called directly from an adapter's `score` function. It requires a non-null `output` and the `expected` value for comparison.

The aggregate score is a weighted mean. Scorer metadata, such as judge rationales, diffs, and validation errors, becomes reflection `feedback`. Individual scores are returned as `objectiveScores` for per-objective Pareto selection.

| Option          | Default                | Effect                                                                 |
| --------------- | ---------------------- | ---------------------------------------------------------------------- |
| `scorers`       | required               | The scorers to run. Each may return a `Score` object or a bare number. |
| `weights`       | `1` per scorer         | Relative weight by scorer name. Must be finite and non-negative.       |
| `buildFeedback` | one line per scorer    | Overrides how scorer output becomes reflection feedback.               |
| `isTransient`   | `false` for all errors | Identifies infrastructure-related scorer failures.                     |

Duplicate scorer names throw because `objectiveScores` can store only one value per name. If `isTransient` classifies a scorer failure as infrastructure-related, the aggregate uses the remaining scorers and is marked `transient`, preventing it from being cached.

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

The decorator works with [`@textopt/ai-sdk`](https://github.com/ctdio/textopt/tree/main/packages/ai-sdk), [`@textopt/langchain`](https://github.com/ctdio/textopt/tree/main/packages/langchain), and custom adapters. It preserves additional methods and generic component names, so wrapping a `GepaAdapter` returns a `GepaAdapter` with the same component type.

Logged events include the candidate, instance feedback, `iteration`, `phase`, `split`, and `candidateId` for grouping rollouts by their place in the optimization run.

When the wrapped adapter reports `usage`, each event carries `metrics` under the names Braintrust reads — `prompt_tokens`, `completion_tokens`, `tokens`, and `cost_usd` — so an experiment shows what a run spent. A reading the adapter did not report is omitted rather than logged as zero.

| Option       | Default        | Effect                                                                            |
| ------------ | -------------- | --------------------------------------------------------------------------------- |
| `adapter`    | required       | The adapter to wrap.                                                              |
| `logger`     | required       | A Braintrust `Experiment`, `Logger`, or another object implementing `log(event)`. |
| `metadata`   | none           | Merged into every event's metadata.                                               |
| `toInput`    | the row itself | Maps a dataset row to the logged input.                                           |
| `toExpected` | omitted        | Maps a dataset row to the logged expected value.                                  |

Logger failures emit a warning and do not fail the rollout.

Types: `BraintrustScorerOptions`, `BraintrustScorerFn`, `BraintrustScorerArgs`, `BraintrustScoreLike`, `BraintrustLoggingOptions`, `BraintrustLoggerLike`, `BraintrustEvent`.

## License

MIT
