---
"textopt": minor
---

`GepaAdapter.evaluate` returns a `ReflectiveBatch` — an `EvaluationBatch` with
`feedback` required. An adapter that returned scores and no prose left the
reflection prompt rewriting instructions from empty feedback blocks, which is
blind search reported as a normal run. Adapters that already return `feedback`
need no change; the rest now fail to compile.
