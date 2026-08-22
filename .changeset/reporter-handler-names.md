---
"textopt": patch
---

`createReporter({ on })` takes a handler map whose keys are checked against the
optimizer's event union, so a misspelled event name is a compile error rather
than a reporter that runs to completion having seen nothing. A reporter built
this way also declares what it handles, and a run warns at start when a handler
names an event that run never emits.
