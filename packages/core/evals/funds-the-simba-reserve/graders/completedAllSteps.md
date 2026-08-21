---
type: regex
weight: 1
pattern: '"completedAllSteps":\s*true'
target: { source: file, path: out/verdict.json }
---
