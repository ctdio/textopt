---
"textopt": patch
---

`createPromptAdapter({ run, score })` from `textopt/gepa` is the single-prompt
case named: `run` receives the candidate's text as `instruction` and `score`
grades the output. It reads which component to run off the candidate and throws
when there is more than one, because a component no module runs is text the
search rewrites every iteration for no effect — that system wants
`createPipelineAdapter`. Neither helper is GEPA-only, and the docs now say so:
a `GepaAdapter` is the base `Adapter` with reflection's evidence added, so the
same adapter passes unchanged to SIMBA, OPRO, MIPRO and both searches.
