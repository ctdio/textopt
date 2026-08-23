---
"textopt": minor
---

A score an adapter synthesized after catching an error is never written to the
evaluation cache. Adapters say so with `failed` on `ScoreResult`, which takes no
judgement about the provider; `transient` still decides what is retried and kept
out of a candidate's mean, and still classifies nothing by default. A run that
finished with failures nothing classified reports how many under the new
`unclassifiedFailures` warning code.

A repeat run over a warm `file-cache` that hit failures now re-runs those
instances rather than reading their zeros back, so it can spend more rollouts
and reach a different winner than the run before it.
