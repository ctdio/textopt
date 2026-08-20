---
"textopt": patch
---

SIMBA's advice proposer sees what each component already says.

`buildAdvicePrompt` named the components it wanted advice for but never showed
their text, while the advice it produces is appended to that text rather than
replacing it. A proposer that cannot read what it is appending to writes blind:
it restates guidance the component already carries, and it cannot correct
guidance that is wrong, since contradicting a line it never saw is not something
it can choose to do. SIMBA's reference implementation passes the current
instructions for exactly this reason.

`AdvicePromptArgs` now carries `current`, a map from component name to what that
component holds, and the prompt renders each as a `<component name="…">…
</component>` block followed by the instruction not to restate advice already
present. A custom `AdvicePromptBuilder` receives the extra field and may ignore
it; anything constructing `AdvicePromptArgs` by hand, or parsing the built
prompt's component list, has to be updated.
