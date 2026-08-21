---
type: regex
weight: 1
pattern: '"metricLoaded":\s*true'
target: { source: file, path: out/verdict.json }
---
