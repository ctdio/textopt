---
"textopt": patch
---

`candidateAccepted` carries `objectiveScores`, the per-objective mean over the
instances the candidate was measured on. A single objective collapsing while the
aggregate holds is how a degenerate metric channel announces itself, and it was
previously visible only in the result.
