---
"textopt": patch
---

A demonstration lands in a component without erasing what else it says.

SIMBA's `appendDemo` rebuilt a demo component from its demos alone, so any text
the component held that was not a demo block was gone the first time a rollout
was harvested into it — including the advice `appendRule` had just written
there. The two mutations could not share a component, which is why
`instructionComponents` defaulted to the components `demoComponents` did not
name, and why a candidate with a single component got one mutation instead of
two. SIMBA's reference implementation appends demos and instructions to the same
predictor; a component is the closest thing this library has to one.

`replaceDemos` rewrites the demo blocks in a text and leaves the rest of it
alone, and both `appendDemo` and the loop's demo-dropping now go through it. A
component named in `demoComponents` can hold instructions too, and the default
`instructionComponents` falls back to every component when every component holds
demos, rather than leaving `appendRule` with nowhere to write and throwing.

Runs where demo and instruction components were already disjoint are unaffected
except that a demo block now keeps its position in the text rather than
replacing it. Runs where they overlapped were losing text and are not
comparable to their old results.
