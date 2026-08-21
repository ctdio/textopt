---
"textopt": minor
---

`GepaOptimizer` refuses an acceptance policy that cannot accept anything at the
configured `minibatchSize`. A sign-flip test over three instances bottoms out at
p = 0.125, so `pairedPermutationAcceptance({ alpha: 0.05 })` at the default
minibatch rejected every proposal and returned the seed after spending the whole
budget. Acceptance policies report the batch they need as `minimumPairs`; raise
`minibatchSize` to at least that, or loosen `alpha`.
