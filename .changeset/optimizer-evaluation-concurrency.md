---
"textopt": patch
---

SIMBA and bootstrapped few-shot search take a `concurrency` option, and OPRO
and random search now apply theirs to evaluation as well as to proposals: a
round's screens or sweeps run together, SIMBA overlaps the candidates a step
built and the finalist sweeps, and a bootstrap candidate's sweep overlaps the
harvest behind it. Every fan-out draws its random stream and prices its
schedule before dispatching, and commits in the order the search proposed, so a
seeded run reaches the same candidates and spends the same rollouts at any
concurrency. Defaults stay at one evaluation at a time.
