# Preparing the data a search runs on

The data decides what a search can find. A metric can be perfect and a budget
generous, and a run will still report a number that means nothing if the rows
are arranged wrong. Do this before the metric pre-flight, because two of the
checks there need splits that already exist.

## The three sets do different jobs

```ts
await optimizer.optimize({
  trainingSet, // reflection reads these: outputs, feedback, what went wrong
  validationSet, // the search selects candidates against these
  testSet, // scored once, at the end, on the winner only
  // …
});
```

`trainingSet` is where evidence comes from. Reflective search mines these rows
for what a candidate got wrong, so a training row earns its place by being
_diagnostic_ — a row every candidate passes teaches the rewriter nothing.

`validationSet` is what selection pressure is applied to, for the whole run.
`bestScore` is a mean over it, which is why `bestScore` is partly fitted and
the result says so.

`testSet` is the only set nothing was ever selected against, and `testScore` is
therefore the only number in a result you can report to someone else. Omit it
and there is no such number — `testScore` is simply absent.

Passing no `validationSet` at all silently reuses `trainingSet`, and the run
carries `validationSetReusesTraining` to say so. That is the right default for
a first look and the wrong thing to report from, because reflection mined the
same rows that then judged the result.

## Split by group, never by row

This is the failure that survives every other check.

Real datasets contain families: the same ticket filed twice, a question and its
paraphrase, five rows generated from one template, transcripts from one long
session. Shuffle and slice, and a family lands on both sides of the boundary.
Reflection then reads one member and the search is selected on its twin, so the
prompt memorises a fact that scores on validation and generalises to nothing.
The gap does not show up as a warning; it shows up as a `testScore` well below
`bestScore`, or worse, not at all if you never held out a test set.

Read how the corpus was built before you infer anything about it. Structure
that is expensive to detect from the rows is often free to read about: a
benchmark that says it ships contrast pairs, a generator with a template count,
a README naming the sampling frame. Deliberate structure is the kind that leaks
worst and the kind most likely to be documented.

Then find the grouping key — a source id, a customer, a template, a document, a
session. If there is no explicit key, near-duplicate text is a usable stand-in: normalise whitespace and case, and cluster on a shared
distinctive phrase or a similarity threshold. Then assign whole groups to
splits. Never assign rows.

Check it afterwards rather than trusting it: no group id appears in two splits,
and the three splits are disjoint and cover the rows you meant to use.

## Every class in every split

A class absent from `validationSet` is a class the search is free to break,
because nothing measures it. A class absent from `testSet` means the final
number says nothing about it.

Rare classes are where this bites, and they are usually the ones that matter —
the refund case, the abuse report, the one legal asked about. Stratify the
split by label so each set holds some of each, and when a class is too small to
stratify, say so out loud rather than letting the split decide silently.

Imbalance also quietly sets the metric's ceiling. If 80% of rows are one easy
class, a candidate that handles only that class scores 0.8, and the 0.8 looks
like progress.

## Sizes

There is no defensible universal number, but the constraints are real:

| Set             | What sets the floor                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------ |
| `trainingSet`   | Enough diagnostic rows that a minibatch shows a candidate failing in more than one way                 |
| `validationSet` | Enough that the mean is not moved by one instance; this is also what a full sweep costs per acceptance |
| `testSet`       | Enough that a difference you would act on is larger than its own noise                                 |

A validation set of 10 makes every instance worth 10% of the score. A search
will find the one instance it can flip.

Cost scales with `validationSet`, not with the other two — see `tuning.md` for
what a sweep costs per optimizer, and note that SIMBA reserves
`min(candidates + 1, maxSteps + 1) × |val|` rollouts before the first step.

## Order of operations

1. Dedup exact repeats. They inflate whichever split they land in.
2. Find the grouping key. Cluster near-duplicates if there is none.
3. Split by group, stratified by label.
4. Verify: groups disjoint across splits, every class present in each.
5. Only now run the metric pre-flight in `metric-preflight.md`, using the
   splits you just made.

## What to do when there is not enough data

Small sets are the normal case, not a failure. What matters is that the
smallness is stated rather than hidden:

- Prefer a real `testSet` over a larger `validationSet`. A fitted number
  measured on more rows is still fitted.
- With too few rows to make three sets, make two and report that `bestScore` is
  all you have — do not pass `testSet` and then report `bestScore` anyway.
- Do not manufacture rows by paraphrasing existing ones into another split.
  That is the group-leakage failure, performed deliberately.
