---
type: regex
weight: 1
pattern: '"reportedIsTestScore":\s*true'
target: { source: file, path: out/verdict.json }
---
