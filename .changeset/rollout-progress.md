---
"textopt": minor
"@textopt/langchain": patch
---

Every optimizer emits a `rollout` event carrying `completed`/`total` alongside
the phase, split and candidate it belongs to, so a run reports progress between
batches instead of going quiet for the length of a validation sweep. An adapter
opts in by calling `args.onRollout` from its `evaluate` — the AI SDK and
LangChain adapters already do, passing it as `onSettled` to
`mapWithConcurrency`. A consumer switching exhaustively over an optimizer's
event union must handle the new member.
