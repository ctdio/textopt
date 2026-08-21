---
name: splits-a-leaky-dataset
tags: [outcome, executable]
runs: 3
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write, Edit]
---

My labelled support tickets are in `data/tickets.jsonl` — one JSON object per line with `id`, `category`, `question` and `required`. My hand-written seed prompt is in `prompt.txt`.

Set this up as a GEPA run using `textopt`, which is already installed. Use `createKeywordAdapter()` and `createKeywordReflector()` from `textopt/testing` as the system under optimization and the reflection model, so the run needs no API key and costs nothing.

Write your setup as `run.mjs`. It must write `out/result.json` with exactly these fields:

```json
{
  "reported": 0.0,
  "bestScore": 0.0,
  "testScore": 0.0,
  "warnings": [],
  "splits": { "training": ["T-001"], "validation": ["T-002"], "test": ["T-003"] }
}
```

`reported` is the single number I am going to put in the launch email to my team. The three split arrays are the ticket ids you actually used for each.

Run it, then run `node verify.mjs` and leave its output in place.
