# Validating a metric before a search runs

Three things have to be true before any number a run reports means anything:

- the splits must not leak — see [Preparing the data](./data-prep.md)
- the metric must have range: it must separate candidates, and move over an
  interval wide enough to see
- its noise must be smaller than the effect you are looking for

They are preconditions, not a checklist. Everything downstream — which
optimizer, which minibatch size, which acceptance test — is reasoning stacked
on an instrument, and reasoning about the instrument is cheaper than every
argument it supports. All three are checkable in an hour, without an optimizer.

The four checks below establish them, in the order to run them. Do not start a
real run until they pass.

## 1. Discrimination on known controls

The highest-value check and the one almost nobody runs.

Write three to five candidates whose ranking you already know:

- the seed
- one deliberately sabotaged — drop the constraint you care most about
- one hand-tuned, as good as you can write by hand
- optionally: an empty instruction, and one that is verbose but wrong

Score each on a slice of the validation set and compare the ranking to your
own. If the metric does not reproduce it, no search will find anything. It
will wander, accept noise, and report a number.

This is also the fastest way to discover reward hacking before the search
does. A verbose-but-wrong candidate that outranks a terse correct one tells you
the judge is paying for length.

## 2. Test-retest variance

Score the _same_ candidate over the _same_ instances three times, with caching
off. The per-instance spread is the metric's own noise.

That number decides two things the defaults cannot decide for you:

- whether `pairedPermutationAcceptance` and `lowerBoundEvaluationPolicy` are
  worth their cost. Turn them on when you have measured large variance, not on
  principle — on the benchmark's pipeline task the pair loses to plain GEPA at
  p = 0.027, because a significance bar on every acceptance is expensive when
  improvements to one component only pay off after another is finished
- how wide a minibatch has to be to say anything. A sign-flip test over n
  instances bottoms out at a p-value of 2^-n

Deterministic metric, zero variance: keep the defaults.

## 3. Spread, not ceiling or floor

Look at the distribution of the seed's per-instance scores, not its mean.

- **Everything at the ceiling.** Nothing to improve; every proposal ties and
  acceptance resolves ties by whatever the noise did. A run reports
  `seedScoreSaturated`.
- **Everything at zero.** Ambiguous. It is what a seed with everything to gain
  looks like _and_ what a broken metric looks like, and the score cannot tell
  them apart — check 1 can. Reported as `seedScoreFloored`.
- **Everything identical in the middle.** The metric is measuring something the
  candidate does not control.

What you want is real per-instance variation. GEPA's frontier is taken over
validation instances, so a seed row with no spread makes the frontier
degenerate before the search starts.

Then ask the same question of the scale itself. A judge's aggregate is bounded
by 0 and 1, but the interval it actually moves in on realistic candidates is
usually narrower, and it is the realized span — not the theoretical one — that
every later number is denominated in. Score your controls from check 1, take
the highest and the lowest, and write the difference down. If the best
candidate you can write by hand scores 1.0 and the sabotaged one scores 0.6,
your instrument has a span of 0.4, and an improvement that reads as "+0.02" is
5% of the available range rather than 2% of it.

A narrow span is not automatically wrong, but it is always worth an
explanation. The common cause is a requirement enforced twice: see the note on
`gate` below.

## 4. Agreement with your own labels

Hand-label 30–50 outputs — good, bad, and the ambiguous middle — then have the
judge grade the same ones and compare.

If the judge cannot reproduce your labels, the search is optimizing the judge's
idiosyncrasies rather than the task. That is still optimization, and the number
will still climb.

Use the disagreements to rewrite the criteria. A criterion the judge and you
read differently is a criterion description problem, not a model problem.

## What to fix, in order

| Finding                                 | Fix                                                                     |
| --------------------------------------- | ----------------------------------------------------------------------- |
| Ranking of controls is wrong            | The criteria. Nothing downstream is worth doing until this passes       |
| Judge rewards length or confident tone  | Add an explicit brevity criterion; make correctness the higher `weight` |
| A hard requirement can be averaged away | Give it a `gate`, and keep its `weight` low — see below                 |
| Realized span is much narrower than 0–1 | Look for a requirement enforced twice, by a `gate` and a heavy `weight` |
| Variance is large                       | Widen the minibatch first, then consider the significance guards        |
| Seed saturated                          | A harder validation set, or a metric that separates these instances     |
| Seed floored and controls do not rank   | The metric is broken, not the seed                                      |
| Judge disagrees with your labels        | Rewrite the criterion descriptions; re-check against the same labels    |

## Enforce a requirement once

`gate` and `weight` are different instruments and it is tempting to reach for
both. A gate makes a requirement non-negotiable: below the threshold the
instance scores 0 whatever else it did. Once a gate is doing that, a heavy
weight on the same criterion buys nothing — every candidate that clears the
gate satisfies the requirement, so the weight is only redistributing score
among candidates that already agree.

What it costs is range. A heavy weight on a criterion nearly every surviving
candidate maxes out pins that share of the aggregate near its ceiling, and the
span left for the search to move in is whatever the other criteria carry.
Gate the requirement, then weight it low, and let the weights go to the
criteria that actually vary.

The general form: when a second mechanism starts enforcing a constraint, check
whether the first still needs to.

## A note on what this is

This is a manual procedure, not a library function. `createJudge` and the
optimizers give you the pieces; the judgement is yours. If you run these checks
often against the same task, the natural thing is to script them — a few
`adapter.evaluate` calls over a fixed batch and a printed table.
