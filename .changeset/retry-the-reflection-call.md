---
"textopt": minor
---

A `reflect` call that throws is retried under the run's existing `retry`
policy, rather than ending the run. Nothing is classified on the way past: a
proposal model is a pure request, so the attempt after a transport failure is
free to succeed and a genuine bug fails `attempts` more times and surfaces
unchanged. Every attempt counts against the optimizer's reflection ceiling —
GEPA's `reflection.maxCalls`, OPRO's and SIMBA's `maxReflectionCalls` — which a
round can now overrun by up to `attempts`. Set `retry: { attempts: 0 }` to keep
the old behaviour. `withRetries(model, policy)` is exported for applying the
same policy to any other `TextModel`.
