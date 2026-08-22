---
"textopt": patch
---

`createPromptAdapter({ run, score })` from `textopt/gepa` is the single-prompt
case named: `run` receives the candidate's text as `instruction` and `score`
grades the output. It reads which component to run off the candidate and throws
when there is more than one, because a component no module runs is text the
search rewrites every iteration for no effect — that system wants
`createPipelineAdapter`.
