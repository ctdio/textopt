# Benchmark

`pnpm bench` runs every optimizer over four offline tasks and twenty seeds, and writes [`bench/results/latest.json`](../bench/results/latest.json). Scores are held-out; `p` is a paired sign-flip test against that task's winner.

| Task           | Winner  | Score | Runner-up         | Score | p     |
| -------------- | ------- | ----- | ----------------- | ----- | ----- |
| `clean`        | `gepa`  | 0.729 | `simba`           | 0.383 | 0.000 |
| `noisy`        | `simba` | 0.525 | `gepa`            | 0.389 | 0.001 |
| `interacting`  | `gepa`  | 0.750 | `randomSearch`    | 0.235 | 0.000 |
| `demonstrated` | `gepa`  | 0.792 | `bootstrapSearch` | 0.598 | 0.000 |

The first three tasks differ only in their metric: `noisy` adds per-instance jitter to the same scoring function `clean` uses, and `interacting` pays only when two components are correct together. The split is the one SIMBA's premise predicts — mining disagreement costs rollouts that a noiseless metric never repays, and pays for itself once the metric is noisy.

`demonstrated` differs in the system instead of the metric: it is right on some instances and wrong on others of the same kind, rather than being a pure function of its prompt. That is the only condition under which a demonstration search has anything to harvest, and it shows in the column — `bootstrapSearch` scores 0.000 on `clean` and `interacting`, where a rollout can only return the candidate's own words, and 0.598 here against a zero-shot 0.333, beating MIPRO, OPRO, SIMBA and random search while calling no proposal model at all. Read its zeroes as the shape of the question rather than as a verdict on the optimizer.

Read these as evidence about the search, not about your task. The proposal model is a deterministic stand-in, so the benchmark holds proposal quality fixed and measures what each search does with it. Use `compare()` on your own task and metric before choosing.
