---
"textopt": minor
---

A harvested rollout cannot end the demo block it is stored in.

Demo blocks are delimited by `<demo>`, and the values inside them were written
raw. A system that quotes its own prompt back produces the one output that
breaks: a rollout worth keeping whose text carries `</demo>`, which closes the
block early and leaves the rest of it as loose text. `parseDemos` then returned
a demo that was not the one stored, and SIMBA reparses and rewrites its demo
components at every step, so the loss compounded over a run instead of showing
up once.

`<demo>`, `<input>` and `<output>` are now escaped in serialized demo values and
unescaped on the way back, so a demo containing a demo round trips as itself.
The escape is escaped first, so a value that already reads `&lt;demo>` survives
too. Demo blocks written by earlier versions still parse; blocks whose values
contain those tags will render them escaped from now on, which changes the text
a component holds and so the candidates a run compares.

A custom `renderDemo` is responsible for its own escaping: the library cannot
know which part of what a renderer emits is the delimiter it meant to write.
