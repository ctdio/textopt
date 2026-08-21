---
"textopt": minor
---

`JudgeCriterion` takes `weight` and `gate`. A gated criterion that grades below
its bar scores the instance 0 whatever the other criteria said, so a hard
requirement is no longer something a search can trade away against three
cosmetic ones. When `expected` is passed, the default judge prompt also forbids
the feedback from restating it — feedback is rewritten into a reusable
instruction, and a fact copied out of the gold answer becomes an answer key
memorised in the prompt.
