---
"textopt": minor
---

`createFileCache` requires a `namespace` naming the system its scores measure,
and never serves an entry written under a different one. A durable log outlives
the model behind an alias, the decoding settings, and the scorer version — every
part of a measurement a cache key does not name. Pass the same string you would
pass as `cacheNamespace`.
