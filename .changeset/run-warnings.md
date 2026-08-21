---
"textopt": minor
---

Every result and `finish` event carries `warnings`: what a run could see about
its own measurement that its numbers cannot say. A run given no `validationSet`
reports that selection ran on the instances reflection read, and a seed the
metric scores identically on every validation instance reports that there was
nothing to rank. Pass `validationSet: "reuseTraining"` to accept the reuse by
name. Custom optimizers implementing `OptimizerResult` or emitting `RunFinished`
must now populate `warnings`.
