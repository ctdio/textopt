---
"textopt": patch
---

`createReporter({ on })` takes a handler map whose keys are checked against the
optimizer's event union, so a misspelled event name is a compile error rather
than a reporter that runs to completion having seen nothing. A reporter built
this way also declares what it handles, and a run warns at start about a
handler named for an event no optimizer emits, or about a reporter whose
handlers all miss the run it is attached to. A handler for an event some other
optimizer emits is not warned about: one reporter written for several searches
is what the shared events are for.
