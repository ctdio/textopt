---
"textopt": minor
---

A comparison reports what each entrant was given, and claims only what its
seeds actually put to the test.

`compare()` reports the work a run got for free. Two entrants held to the same
`maxMetricCalls` do not do the same amount of work: one that revisits candidates
it has already scored is served from the cache, and one that never revisits pays
for every rollout, so ranking them on `metricCalls` alone credits the difference
to the search. Every run now carries `cacheHits` and `reflectionCalls`, summarized
as `meanCacheHits` and `meanReflectionCalls`, and reflection calls are counted
because no metric budget covers them — an entrant can be cheap on rollouts and
expensive on proposals. `OptimizerResult` carries both: `cacheHits` on every
optimizer, `reflectionCalls` on the five that propose text. Anything implementing
that interface by hand has to report `cacheHits`.

A p-value is withheld where the seeds never earned one. `compare()` reported
`pValueVsWinner` for every non-winner, including entrants whose margin over the
winner was identical at every seed — a deterministic search against a
deterministic model, or one whose seed reaches nothing that varies. A sign-flip
test over n seeds answers a question about n independent trials; n copies of one
realization is a single trial reported as n, and the p-value it produces goes as
low as 2^-n while resting on nothing. Those comparisons now report no p-value at
all, and `distinctScores` says how many distinct outcomes an entrant actually
had, so a row of identical scores is visible rather than inferred. A margin of
exactly zero at every seed is unaffected: p = 1 claims nothing and is honest.

`pValueVsWinnerHolm` corrects for the family the raw p-value is read against.
Comparing six entrants against a winner runs five tests, and the smallest of five
is smaller than one test's worth of evidence. Holm-Bonferroni step-down adjusts
each surviving comparison against the whole family — including the slots held by
comparisons withheld above, which are still members of it.

Exact enumeration now reaches twenty seeds rather than sixteen. `signFlipPValue`
builds the reachable sums by doubling instead of re-summing each of the 2^n sign
masks, which costs O(2^n) rather than O(2^n · n) — about four times faster at
n = 20, 16ms against 63ms. Comparisons between sixteen and twenty seeds get the
exact p-value where they previously fell back to the normal approximation.

Random search starts a round whenever one more sweep is affordable. It required
the whole round up front — every variant proposed and scored — which stranded up
to `variants * |validationSet| - 1` rollouts unspent at the end of a run, around
a fifth of a typical budget at the default settings. A variant needs only its own
sweep to be compared against the incumbent, and the round already truncates to
what the remainder can fund whole sweeps for. The cost is bounded and paid once:
the final round proposes up to `variants - 1` texts it never scores. Runs at a
fixed `maxMetricCalls` will evaluate more variants than before and can select a
different candidate.
