# Benchmark

`pnpm bench` runs every optimizer over four offline tasks and twenty seeds, and writes [`bench/results/latest.json`](../../../bench/results/latest.json).

Each task is a support-ticket policy. A ticket has four features — `tier`, `channel`, `issue` and `region` — and a hidden policy of nine rules says which actions a correct answer must take. The system under optimization reads rules out of the candidate and applies them to the ticket, so a candidate is scored on what it made the system do rather than on the words it contains. No rule keys on `region`: it is there to be mistaken for a reason, and a search that believes it pays for the belief.

All seventy-two feature combinations are dealt into three disjoint splits of twenty-four. The splits are balanced, not merely disjoint — the same mix of every feature value in each — because every search here selects on the validation set and is reported on the test set, and two splits that ask for different things turn a held-out score into a lottery. No pair of features is in lockstep within a split either, so no rule can hide behind another. A candidate that fitted the combinations it was shown scores zero; one that found the rules scores the same on tickets it never saw.

Held-out score, twenty seeds:

| entrant             | `clean`   | `noisy`   | `interacting` | `demonstrated` |
| ------------------- | --------- | --------- | ------------- | -------------- |
| `gepa`              | **0.947** | 0.920     | **0.891**     | 0.894          |
| `gepaVarianceAware` | 0.945     | **0.931** | 0.835         | **0.910**      |
| `opro`              | 0.285     | 0.283     | 0.170         | 0.693          |
| `simba`             | 0.249     | 0.282     | 0.185         | 0.657          |
| `randomSearch`      | 0.266     | 0.268     | 0.111         | 0.646          |
| `mipro`             | 0.094     | 0.140     | 0.085         | 0.649          |
| `bootstrapSearch`   | 0.000     | 0.060     | 0.000         | 0.740          |

Read these four rows first. They are what the table has to be judged against:

| reference   | `clean` | `noisy` | `interacting` | `demonstrated` | what it is                                |
| ----------- | ------- | ------- | ------------- | -------------- | ----------------------------------------- |
| `policy`    | 1.000   | 0.945   | 1.000         | 1.000          | the rules being searched for; the ceiling |
| `bestFixed` | 0.400   | 0.409   | 0.400         | 0.583          | the best answer that ignores the ticket   |
| `shotgun`   | 0.358   | 0.378   | 0.358         | 0.358          | every action on every ticket              |
| `zeroShot`  | 0.000   | 0.060   | 0.000         | 0.625          | the seed candidate, unoptimised           |

`bestFixed` and `shotgun` are the floors a search has to beat to have found anything. Neither conditions on the ticket at all: `shotgun` sprays all nine actions and eats the bloat penalty for the ones that miss, and `bestFixed` is the best subset of actions to spray, chosen over the training and validation sets and never over the held-out set it is reported on — a floor selected on the number it is published at is an oracle, not a floor. On the three rule-shaped tasks the two GEPA rows are the only entrants that clear either floor. Every other search — `opro`, `simba`, `randomSearch`, `mipro`, `bootstrapSearch` — finishes below a candidate written without searching at all, having spent its whole budget to get there. On `demonstrated` all seven clear the floors, because the seed candidate already does.

## What the diagnosis is worth

Every entrant is run a second time with exactly one thing removed: the metric's per-instance diagnosis is stripped from its prompt, leaving it the blind draw a score-only search already gets. Its settings are the ones it was tuned at, and its proposals are capped at nine — the fewest any entrant makes when left alone — because blind a proposal is a draw from a fixed pool rather than an induction, and an entrant that buys more draws scores better without searching better.

| entrant             | `clean`      | `noisy`      | `interacting` | `demonstrated` |
| ------------------- | ------------ | ------------ | ------------- | -------------- |
| `gepa`              | 0.185 −0.762 | 0.179 −0.741 | 0.089 −0.802  | 0.656 −0.238   |
| `gepaVarianceAware` | 0.059 −0.885 | 0.121 −0.811 | 0.022 −0.813  | 0.631 −0.279   |
| `opro`              | 0.097 −0.188 | 0.134 −0.149 | 0.065 −0.105  | 0.647 −0.045   |
| `simba`             | 0.137 −0.112 | 0.169 −0.113 | 0.087 −0.097  | 0.659 +0.002   |
| `mipro`             | 0.094 0.000  | 0.140 0.000  | 0.085 0.000   | 0.649 0.000    |
| `randomSearch`      | 0.266 0.000  | 0.268 0.000  | 0.111 0.000   | 0.646 0.000    |
| `bootstrapSearch`   | 0.000 0.000  | 0.060 0.000  | 0.000 0.000   | 0.740 0.000    |

The blind score first, then what redaction cost against the table above.

This is the largest effect in the benchmark, and it is not a fact about search. Blind at nine proposals apiece the entrants land in a band — 0.022 to 0.266 on the three rule-shaped tasks — and the winner of that band is `randomSearch`, which does not search reflectively at all. `gepa` blind scores 0.185 on `clean` against its own 0.947. Whatever separates these algorithms when they can read the diagnosis, almost none of it survives when they cannot.

The bottom three rows lose exactly nothing, which is the check that the redaction is measuring what it claims to. `bootstrapSearch` calls no proposal model, and neither `mipro` nor `randomSearch` was ever shown a per-instance diagnosis to begin with — so redacting one changes nothing about what they were given, and their scores are identical to the digit. Those are also the only rows whose proposals the cap does not reach: `mipro` proposes its menu up front, seventeen entries on `interacting`, and `randomSearch` takes sixteen to twenty. The cap binds on every entrant whose score moves at all.

What the middle rows show is that reading the diagnosis is necessary but not sufficient. `opro` and `simba` both get it and both lose real ground without it — 0.188 and 0.112 on `clean` — but that is a fifth of what `gepa` loses, because they were converting much less of it into rules in the first place. So the column separates three things a single table cannot: what an entrant is given, what it does with it, and what it would score on neither.

Anyone choosing an optimizer from the table above should read this one alongside it. Most of what the top rows are buying is a proposal step that reads what the metric said about individual tickets. That is a real difference between these algorithms — what each one asks its model for is part of the algorithm — but it is a difference in the question they ask, not in how well they search the answers.

## The tasks

`clean` is the reference case: a noiseless metric with a clean gradient.

`noisy` adds per-instance jitter to the same metric, which is why its ceiling is 0.945 rather than 1.000 — noise costs even a perfect candidate something. It is the one task where `gepaVarianceAware` wins, which is what its acceptance test is for.

`interacting` splits the policy across two components in a pipeline: `triage` holds the rules about the issue, `response` those about the tier and channel, and the system applies `response` only to tickets `triage` already handled correctly. Every improvement to the second component scores nothing until the first is complete, so the task measures where a search spends. It has the widest spread between seeds of any task here (`gepa` sd 0.109), which is what a search that can be sent down the wrong component looks like, and it is the one task where the two GEPA rows genuinely separate.

`demonstrated` changes the system rather than the metric — it answers a share of tickets correctly with no help from its prompt, which is the only condition under which harvesting rollouts has anything to harvest. It is also the task where the unoptimised seed already scores 0.625, and where the entrants that search instructions barely move off it: `mipro` 0.649, `randomSearch` 0.646. `bootstrapSearch` reaches 0.740 while calling no proposal model at all. When a system's failures are inconsistency rather than instruction, searching the instruction is the wrong tool, and this row is what that costs.

It is the one task where `simba` harvests. SIMBA has two mutations — append a rewarded rollout as a demonstration, or ask a model what the better run did and append that as a rule — and which of the two it draws from is tuned like any other setting. Demonstrations are selected here and nowhere else, worth 0.657 against 0.628 for rules alone, and the three tasks above select rules, where a demonstration spends a draw on a rollout that carries nothing its prompt did not.

## Two rows that are not what they look like

`bootstrapSearch` scores 0.000 on `clean` and `interacting`, the only entrant with no seed spread at all on them. There is nothing to harvest there — the answer on those tasks is a pure function of the candidate, so a rollout tells the search nothing its prompt did not, and a demonstration search can only hand a candidate its own words back. It is run on all four tasks rather than only where it wins precisely so that row is visible.

`mipro` scores 0.094 on `clean` and 0.085 on `interacting`. Its menu is proposed once, up front, from a dataset summary and a sample of task inputs — nine reflection calls on `clean`, not one of which carries a per-instance diagnosis. That is MIPROv2's grounded proposer working as designed, and it goes deeper than the stand-in: this benchmark's training data carries no gold labels at all, since what a correct answer requires lives in the metric rather than in the dataset. A real model in MIPRO's up-front slot would have nothing linking features to actions either. On this substrate `mipro` measures propose-before-evidence, and the blind column above puts a number on that: redacting the diagnosis costs it exactly nothing on all four tasks, because it was never shown one.

## How the numbers are produced

Every entrant gets the same `maxMetricCalls` on a task, the same data and the same proposal model. The held-out set is evaluated once, on the candidate the search already chose, outside the budget — so no optimizer can spend rollouts on the number it is reported at.

Settings are tuned per entrant per task over comparable grids, on seeds 100–109, choosing on the validation score alone. The reported score comes from seeds 0–19. Nothing about how an entrant was run is fitted to the seeds or the split it is scored on; `tuning` in the JSON records what each one was given. An earlier version of this benchmark swept one entrant's hyperparameters against the published metric and left the others at a default, which is enough on its own to decide a table.

`p` is a paired sign-flip test against the winner, Holm-adjusted across the entrants in the same comparison. It is withheld where every seed produced the same margin: a test over twenty identical differences reports a precision that twenty runs of one realization never earned. `distinctScores` says how many distinct outcomes an entrant actually had — `randomSearch` is 1 everywhere, since it takes no seed and is deterministic against a deterministic model.

The two GEPA rows separate on one task out of four. Holm-adjusted, the winner's margin over the other runs p = 0.027 on `interacting`, and 0.334, 0.088 and 0.125 on `clean`, `noisy` and `demonstrated` — so only `interacting` clears 0.05, where plain GEPA wins. Every other margin in the table clears it at p < 0.0001. Read the top two rows as one result with two spellings everywhere except `interacting`.

## What this does and does not tell you

The proposal model is a deterministic stand-in. It induces a rule only when a feature value and a missing action co-occur across at least two tickets in its prompt, and otherwise draws from the hundred and eight rules the language can express, of which nine are correct.

That stand-in has one property no real model has: it is exact counting, so it sharpens with the size of its reflection batch and stops being wrong at all past roughly a dozen observations. At that point tuning stops selecting a search and starts selecting whichever entrant asks for the biggest prompt. The grid is therefore bounded at a batch of nine, where the proposer is right about four times in five, and this bound is load-bearing — every minibatch-shaped entrant tunes straight to it. That is a modelling decision to keep the comparison about search, and it is the first thing to be suspicious of if these numbers ever look too clean.

So these numbers are about search behaviour under a fixed, weak proposer. They are not about what any of these optimizers will do with a real model that can read a dataset and write a sensible instruction unprompted. Read the table as evidence about the searches, read the blind column, `bestFixed` and `shotgun` as the scale it should be read at, and then run [`compare()`](./evaluation.md#comparing-optimizers) on your own task and metric — which is the only measurement that answers the question you actually have.
