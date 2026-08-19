---
"textopt": minor
"@textopt/langchain": minor
---

First public release: GEPA, SIMBA, OPRO, MIPRO, bootstrapped few-shot search,
and random search behind a shared optimizer interface, with a LangChain
adapter. The shared substrate handles budgets in rollouts, dollars and wall
clock, transient-failure retry, durable caching, checkpoints and resume for
every optimizer, a model-graded judge, a multi-module pipeline adapter, and
`compare()` for deciding between two runs with a p-value rather than a hunch.
A run reports through `reporters`: any number of observers, each with an
`onEvent` called synchronously and a `flush` awaited as the run ends. GEPA
announces every accepted candidate — the seed included — with the text that
scored and the row it put on the frontier, and reports the held-out sweep per
instance rather than as a lone mean.

The Vercel AI SDK, Braintrust and LangSmith packages are still in beta and ship
from the repository only.
