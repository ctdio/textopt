---
"textopt": patch
---

Two reporters ship with the library: `consoleReporter({ level })` prints one
line per event — `quiet` for acceptances and the finish, `verbose` for
everything — and `jsonlReporter({ path })` from `textopt/file-reporter` appends
each event as a JSON line, leaving structured data structured instead of
flattening it into log prose.
