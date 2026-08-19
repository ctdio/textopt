# Distilling a run

Prompt optimization makes a prompt better partly by making it longer, and you
pay for that length on every inference forever. Distillation is what makes the
length free: run the optimized candidate on the strong model, keep the rollouts
the metric rewarded, and train a smaller model on them. The text moves into
weights.

textopt collects and serializes the training data. It never trains anything —
that is a provider's job, and the providers change.

## The order matters

Distill before optimizing and you freeze whatever your first draft happened to
do. The search is what finds the behaviour worth freezing.

["Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better
Together"](https://arxiv.org/abs/2407.10930) goes further: alternating the two
beats weight optimization alone by up to 60% and prompt optimization alone by
up to 6%. The second prompt pass is an ordinary `optimize()` call with an
adapter pointed at the fine-tuned model, so nothing here is a one-way door —
provided the render step below leaves a prompt to optimize.

## Harvesting

`harvestRollouts` runs a candidate over data and keeps what the metric
rewarded. It is the same primitive `harvestFewShotExamples` uses, without the
four-example ceiling:

```ts
import { harvestRollouts } from "textopt";

const harvest = await harvestRollouts({
  adapter,
  candidate: result.bestCandidate, // the run's winner
  data: unlabeledPool,
  minScore: 0.9,
  maxMetricCalls: 5000,
});

harvest.rollouts; // { input, output, score }[]
harvest.attempted; // instances run, including the ones that failed the bar
```

Filtering on `minScore` is what makes this rejection sampling rather than
imitation: you are copying the strong model on the occasions your metric says
it was right. A boolean metric makes `minScore: 1` the only sensible threshold;
a graded metric asked for a perfect score throws away every rollout that was
most of the way there, which on a hard task is all of them.

**Sweep the right data.** Not the validation set — that is the set that
selected the winning candidate, so its rollouts are enriched for the
candidate's fit to those particular instances rather than to the task, and
there are only ever tens to low hundreds of them. Use the training set, or
better, a pool held out of the run entirely. Volume comes from the sweep, not
from the run: a search bounded at 150 metric calls is a seed, and 5,000
unlabeled instances is a dataset.

The harvest carries its own budget. It is a separate spend from the run that
produced the candidate, and `maxMetricCalls` bounds it.

## Rendering

`toTrainingJsonl` serializes rollouts as one chat-messages example per line —
the shape Axolotl, Together, Fireworks and the Hugging Face trainers all
ingest. Your `render` callback decides what each example looks like, because
only you can read a `Datum`:

```ts
import { toTrainingJsonl } from "textopt";

const jsonl = toTrainingJsonl({
  rollouts: harvest.rollouts,
  render: ({ rollout }) => ({
    messages: [
      { role: "user", content: rollout.input.question },
      { role: "assistant", content: String(rollout.output) },
    ],
  }),
});
```

Return `null` from `render` to drop a rollout. Write the string yourself —
nothing here touches the filesystem.

### How much of the prompt to leave in

This is the decision that matters, and there is no default worth picking for
you.

**Drop it entirely.** The input is the bare instance; the whole optimized
candidate moves into weights. Maximum compression, and it works when the task
is inferable from the input distribution. It fails when the task specification
lives in the prompt — a custom label taxonomy, an output schema, a routing
rubric — because the student then has to reverse-engineer that from examples.
It also ends the alternating loop above: there is no prompt left to optimize.

**Keep a short task statement.** Train on a brief instruction plus the
instance, serve with the same brief instruction. What you distill away is the
optimized delta — the tips and examples the search accreted — rather than the
statement of the job. This is the right default for most systems.

**Keep the full optimized candidate.** No compression at all, which is the
point: the student learns under the prompt it will be served with, and the next
optimization pass tunes that prompt against the new weights. This is the
BetterTogether shape.

## Checking it worked

The number that matters is not the training loss. Point an adapter at the
fine-tuned model and run both against the same held-out `testSet` — data that
neither the prompt search nor the fine-tune ever saw:

```ts
import { compare } from "textopt";

const comparison = await compare({
  seeds: [0, 1, 2, 3, 4],
  entrants: {
    teacher: () => evaluatePrompted(teacherAdapter, result.bestCandidate),
    student: () => evaluateDistilled(studentAdapter),
  },
});
```

`compare()` ranks on `testScore` and reports a paired sign-flip p-value against
the winner, so "the distilled model matches the prompted one" is a measurement
rather than an impression. See [Measuring a result](evaluation.md).
