---
"@textopt/langchain": patch
---

The adapter marks a run or scoring failure it caught with `failed`, so the zero
it stands in for is never written to the evaluation cache. `isTransient` still
decides only what is worth retrying.
