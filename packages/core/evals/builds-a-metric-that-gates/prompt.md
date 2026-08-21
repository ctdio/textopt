---
name: builds-a-metric-that-gates
tags: [outcome, executable]
runs: 3
max_turns: 30
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write, Edit]
---

I am building the metric that will drive an automatic prompt search over my support agent's replies. A grader already produces five raw grades per reply, each an integer from 0 to 5:

- `policy` — 5 means every refund window the reply stated is in the policy document. Anything below 5 means it stated one that is not. **Legal will not ship a system that invents refund windows.** This one is not negotiable.
- `greeting` — the reply opens with a greeting.
- `brevity` — the reply stays under 120 words.
- `warmth` — the tone.
- `ticketId` — the reply mentions the ticket ID. I want this on a dashboard so I can decide later whether I care. It must not influence the score the search optimizes against, at all.

Turn those five grades into the single number the search maximizes.

Write `metric.mjs` exporting `aggregate(grades)`, where `grades` is an object with those five integer keys and the return value is the score for that reply. `textopt` is installed if you want to look at how it treats criteria.

Then run `node verify.mjs` and leave its output in place.
