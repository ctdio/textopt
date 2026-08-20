# Measuring a result

Two questions a finished run cannot answer about itself: how much of its score
is fitted to the set that picked the winner, and whether it beat another
optimizer by more than noise.

## Held-out evaluation

The optimizer selects candidates against `validationSet`, so `bestScore` is fitted to that set and may overstate performance on unseen data.

Pass a `testSet` and the winner is scored on it once, after the search is over:

```ts
const result = await optimizer.optimize({
  seedCandidate,
  trainingSet,
  validationSet,
  testSet, // never seen by the search
  adapter,
  reflect,
  maxMetricCalls: 300,
});

result.bestScore; // on the validation set — the search selected for this
result.testScore; // on instances no candidate was ever selected against
result.testMetricCalls; // charged separately, not against maxMetricCalls
result.testUsage; // and costed separately, outside maxCostUsd
```

The gap between `bestScore` and `testScore` estimates validation overfitting. Test rollouts are reported separately and are outside every ceiling: they do not count against `maxMetricCalls`, and their tokens are in `testUsage` rather than `usage`, because the sweep runs after the search has already stopped. Budget for it the way you would budget for one full validation sweep. The resume fingerprint ignores `testSet`, so it can be added when resuming a run.

All optimizers expose these fields through `OptimizerTask` and `OptimizerResult`.

## Comparing optimizers

A difference in means over a handful of seeds is usually noise. `compare()` runs each entrant over the same seeds, ranks them on `testScore` where a run reports one, and reports a paired sign-flip p-value against the winner:

```ts
import { compare } from "textopt";

const comparison = await compare({
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  concurrency: 4,
  entrants: {
    gepa: ({ seed }) => new GepaOptimizer({ seed }).optimize(task()),
    opro: ({ seed }) => new OproOptimizer({ seed }).optimize(task()),
  },
});

comparison.winner; // highest mean score
comparison.summaries; // mean, sd, min, max, rollouts, cost, pValueVsWinner
comparison.runs; // every individual run
```

Entrants are functions of a seed, not optimizer instances: the seed is constructor config, and every optimizer here is deterministic given one, so comparing two entrants at a single seed compares two anecdotes. Build a fresh task inside each entrant — a shared reflection model with internal state would make each result depend on the runs before it.

Ranking on `testScore` matters. The validation score is the number the search selected against for its whole run, so an entrant that overfits looks strongest on exactly the number it fitted.
