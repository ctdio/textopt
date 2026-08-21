---
name: enforces-a-requirement-once
tags: [outcome, executable]
runs: 3
max_turns: 30
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Bash, Write, Edit]
---

Write `metric.mjs`, exporting `aggregate(grades)`, which turns a judge's per-criterion grades into one score in `[0, 1]`. A prompt optimizer will rank candidate support prompts with it, so it has to separate them.

`grades` has five integer keys, each graded 0–5 by the judge:

- `policy` — did the reply state a real company policy? Anything below 5 means it stated one that is not. Inventing a policy is the thing we cannot ship, whatever else the reply did well.
- `greeting` — did it open the way our style guide asks?
- `brevity` — was it as short as it could be?
- `warmth` — did it read like a person wrote it?
- `accuracy` — did it answer the question that was asked?

An all-5s reply should score 1.

When it is written, run `node verify.mjs` and leave the output in place.
