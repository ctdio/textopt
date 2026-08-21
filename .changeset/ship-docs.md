---
"textopt": patch
---

The long-form guides ship in the tarball, under `docs/`, so an installed copy
documents the version installed rather than whatever `main` has become. Two of
them are new: `data-prep.md` on splitting a dataset a search can be trusted
with, and `metric-preflight.md` on checking a metric separates candidates —
and moves over a wide enough interval — before a budget is spent on it.

Doc comments on the API carry the traps that belong beside the code: that
`weight: 0` removes a criterion from the aggregate but not from
`objectiveScores`, that a `gate` and a heavy `weight` on the same criterion
enforce a requirement twice and narrow the range a search has left to move in,
and that a SIMBA run has to be funded past its finalist reserve before any step
happens.
