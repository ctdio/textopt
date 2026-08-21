---
"@textopt/langsmith": minor
---

`@textopt/langsmith` ships to npm. `createLangSmithReporter` writes any optimizer's run to LangSmith as one experiment per accepted candidate, and matches LangSmith's `Client` structurally, so it adds no runtime dependency on the `langsmith` SDK.
