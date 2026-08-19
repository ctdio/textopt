---
"textopt": patch
"@textopt/langchain": patch
---

First public release: GEPA, SIMBA, OPRO, MIPRO, bootstrapped few-shot search,
and random search behind a shared optimizer interface, with a LangChain
adapter. The shared substrate handles budgets in rollouts, dollars and wall
clock, transient-failure retry, durable caching, checkpoints and resume for
every optimizer, a model-graded judge, a multi-module pipeline adapter, and
`compare()` for deciding between two runs with a p-value rather than a hunch.
Every run reports through `reporters`: any number of observers, each with an
`onEvent` called synchronously and a `flush` awaited as the run ends, including
when it ends by throwing. Each search emits its own event union, but every one
of them announces an accepted candidate with the text that scored and its row
over the validation set, and reports the held-out sweep per instance rather
than as a lone mean. A reporter that reads only those narrows with
`isCandidateAccepted` and `isRunFinished` and drops into any optimizer.

The Vercel AI SDK, Braintrust and LangSmith packages are still in beta and ship
from the repository only.
