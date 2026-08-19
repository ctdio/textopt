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
The Vercel AI SDK and Braintrust adapters are still in beta and ship from the
repository only.
