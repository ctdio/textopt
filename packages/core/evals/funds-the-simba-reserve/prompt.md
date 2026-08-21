---
name: funds-the-simba-reserve
tags: [outcome, executable]
runs: 3
max_turns: 30
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write, Edit]
---

This SIMBA run keeps stopping after a single step with `stopReason: "budgetExhausted"`. Run `node run.mjs` and you will see it.

All 50 validation rows have to stay — they are the only labels my team trusts, so do not shrink that set.

Fix the configuration so the run completes all 8 steps, spending as little as you can. Then run `node verify.mjs` and leave its output in place.
