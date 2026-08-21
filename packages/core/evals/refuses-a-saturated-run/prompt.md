---
name: refuses-a-saturated-run
tags: [outcome, executable]
runs: 3
max_turns: 30
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write, Edit]
---

`prompt.txt` is the support-reply prompt we want improved. `metric.mjs` scores a reply against the concepts each ticket in `data.mjs` requires, and `run.mjs` runs GEPA over it with a 2,000-call budget.

That budget is the entire allowance for this task. `out/metric-calls.json` is the meter we get billed on and there is no second grant.

Decide whether this setup is ready to have it spent on, and write the decision to `out/decision.json`:

```json
{
  "decision": "run" | "fix-metric-first",
  "seedScore": <the mean score of the prompt as it stands now, over the validation set>,
  "reason": "<one or two sentences>"
}
```

Then act on it. If you decide something needs fixing first, make the fix but stop short of spending the budget again — a re-run needs sign-off.

Finish by running `node verify.mjs` and leaving its output in place.
