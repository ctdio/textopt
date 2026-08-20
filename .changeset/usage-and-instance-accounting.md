---
"textopt": patch
---

Rollout accounting survives a resume, and the numbers a run reports name the
candidate it returns.

`maxCostUsd` is a ceiling on the run rather than on the segment: every snapshot
now carries the usage already spent, and a resumed run folds it back in instead
of restarting its token and dollar totals at zero. Harvesting is part of that
total — `harvestRollouts` and `harvestFewShotExamples` report the `usage` their
own evaluator spent, and MIPRO and bootstrapped few-shot search absorb it
through `Evaluator.absorbUsage`, so the pass that collects demos is no longer
invisible to a cost ceiling that is supposed to bound it.

SIMBA keys a minibatch by dataset row instead of by position within the batch.
Two steps drawing different rows shared the ids `0..n-1`, so a candidate that
recurred across steps could be served a cached score another instance had
measured. Its default `instanceId` hashes the datum, as bootstrapped few-shot
search's now does and as the other four optimizers already did.

OPRO sweeps the held-out set with the candidate it returns. A run that screens
on a `scoringSetSize` subset can end on an incumbent the closing full sweep
never confirmed, and `testScore` described that unreturned candidate rather than
`bestCandidate`.

Candidate ids continue across a resume in OPRO, MIPRO, bootstrapped few-shot
search and random search. Reporters key rows by `candidateId`, and a counter
that restarted at zero made a resumed run overwrite the run it continued.

`SAMPLING_POOL` is exported from `textopt/testing`, which the README had
documented but the module did not export.
