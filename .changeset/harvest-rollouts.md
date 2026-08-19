---
"textopt": patch
---

`harvestRollouts` runs a candidate over data and returns the rollouts the
metric rewarded, without the demo ceiling `harvestFewShotExamples` applies, and
`toTrainingJsonl` serializes them as chat-messages JSONL for distilling a run
into a smaller model. `harvestFewShotExamples` keeps its behaviour and now
calls the same primitive; [Distilling a run](docs/distillation.md) covers which
data to sweep and how much of the optimized candidate to leave in the training
input.
