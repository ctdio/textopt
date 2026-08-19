# Working in this repository

textopt is a TypeScript library for optimizing the text inside an LLM system —
prompts, instructions, few-shot blocks. It implements six published search
algorithms over a shared substrate, plus adapters for common frameworks.

Read this before changing anything. The conventions below are not stylistic
preferences; most of them exist because breaking them has already cost someone
a day.

## Layout

```
packages/core        the substrate and every optimizer
  src/                   "."         evaluation, budget, cache, demos, judge,
                                     compare, deadline, checkpoint, rng, types
  src/gepa/              "./gepa"    reflective evolution over a Pareto frontier
  src/simba/             "./simba"   mini-batch ascent on program disagreement
  src/opro/              "./opro"    score-history meta-prompting
  src/mipro/             "./mipro"   joint search with a TPE surrogate
  src/bootstrap-search/  "./bootstrap-search"   few-shot search, no proposal model
  src/random-search/     "./random-search"
  src/file-cache.ts      "./file-cache"         the only entry point using node:fs
  src/testing.ts         "./testing"            fixtures shared by tests
packages/ai-sdk, packages/braintrust, packages/langchain    adapters
examples/                                                   runnable scripts
bench/                                                      offline seed sweeps
```

Each optimizer owns its directory and exports through its own entrypoint. The
substrate must stay free of anything shaped like a particular optimizer — if a
type only makes sense for reflective search, it belongs in `gepa/`.

## Commands

Run everything from the repository root. `cd` inside a compound shell command
persists between calls and will silently point test globs at the wrong place.

```bash
npx vitest run 2>&1 | tee /tmp/test.log     # always capture, then read the file
npx vitest run packages/core/src/mipro      # one directory
npx vitest run -t "name of the test"        # one test
pnpm typecheck                              # builds first, then tsc --noEmit
pnpm format                                 # prettier over the repository
pnpm lint:packages                          # publint + attw on the built packages
pnpm bench                                  # builds, then sweeps every optimizer
pnpm changeset                              # record a release note for a change
```

Never re-run a suite just to re-read a failure. The output is already in the
log file.

`lint:packages` checks the published surface — subpath exports, dual type
declarations — which the tests cannot see, because they import through source.
It needs a build, so it runs after `pnpm build`.

## Releasing

Changesets owns versions. Add a changeset with the change that needs one; never
edit a `version` field or write a release commit by hand. On `main`, the release
workflow opens a `chore: version packages` pull request, and merging that pull
request publishes to npm.

`@textopt/ai-sdk` and `@textopt/braintrust` are `private` and listed in
`.changeset/config.json` under `ignore`, so they neither version nor publish.
Removing both markers is what promotes one out of beta.

## Fidelity to the papers

This library's value is that each optimizer does what its paper says. That puts
two obligations on any change to search behaviour.

**Verify against the reference, not against memory.** Defaults in this codebase
have been wrong because someone recalled them confidently. Fetch the actual
source before asserting what a reference does:

- GEPA — `github.com/gepa-ai/gepa`, paper arXiv 2507.19457
- MIPROv2 — `dspy/teleprompt/` in `github.com/stanfordnlp/dspy`, and Optuna's
  `optuna/samplers/_tpe/sampler.py` for the surrogate's defaults
- OPRO — `github.com/google-deepmind/opro`, `opro/optimization/opt_utils.py`

**Every deviation is deliberate and documented where the option lives.** A
departure from the reference gets a comment saying what the reference does, why
this differs, and what it costs — next to the code, not only in the README. If
you cannot articulate the cost, you do not yet understand the deviation.

## Claims in documentation are measured

The READMEs quote real numbers ("0.87 mean best, 8 runs in 15"). Those came from
sweeps, not intuition. Two rules follow:

- **Measure across seeds.** A single-seed result has twice overturned a
  conclusion that looked obvious in this repo. Sweep 10–30 seeds and compare
  means before claiming an approach is better.
- **Re-measure when defaults change.** Changing a default silently invalidates
  every number measured under the old one. Re-run the sweep and update the
  figures, or delete them.

`bench/` is where sweeps live. It runs every optimizer against the same offline
tasks across seeds, with no network and no model: `bench/src/tasks.ts` supplies
a simulated proposer that reads whichever evidence a prompt carries — written
feedback, a score history, or neither — so what a run measures is the search
rather than the prompt it happened to send.

```bash
pnpm bench          # 3 tasks x 5 configurations x 10 seeds, ~2 minutes
```

It rewrites `bench/results/latest.json`, which is committed: a diff there is the
review artifact for any change to search behaviour. Commit the regenerated file
with the change that moved it.

The tasks are tuned to sit off both ceilings — an optimizer that scores 1.000 on
every seed is measuring nothing. If a change makes a task saturate, widen the
task rather than keeping the number.

Sweeps do not belong in the test suite; a focused regression test pinned to one
seed does.

## Tests

Red/Green/Refactor, without exception. Write the failing test, watch it fail for
the reason you expect, then implement. A test that passes before the change is
not evidence of anything.

- **No mocks.** Not for adapters, not for the optimizers, not for anything owned
  here. Tests drive real optimizers against small in-line adapters. If something
  seems to need a mock, the seam is wrong.
- **Deterministic.** Everything is seeded; no `Math.random()` or wall-clock in a
  test. Vary a run by passing a different `seed`.
- **No conditionals.** No `if`, ternary, or `switch` in a test body — they hide
  which path ran. A fixture's scoring function may branch; assertions may not.
- **Name the behaviour**, not the function: `"retries a rejected text once
another component has moved"`.

When a test's threshold encodes a measurement, put both numbers in a comment —
what it is with the change and what it was without — so the next person can tell
a real regression from a re-tuned constant.

## Code style

File order is fixed: imports, types, constants, main exports, then helpers at
the bottom. The public API of a file should be readable in its first 30 lines.

- `function` at module level, never arrow functions
- More than two arguments means a single object parameter
- `err` in catch blocks, never `e` or `error`
- No `any`; use `unknown` with a type guard
- Comments explain _why_, never _what_ — no narration of what the code plainly
  does, no "we" or "let's"

Match the surrounding code when it disagrees with this. Never mix two styles in
one file.

## Commits

[Conventional Commits](https://www.conventionalcommits.org). The subject is
lowercase after the type and describes what the commit does:

```
feat(mipro): model menu components jointly in the surrogate
fix(opro): scope proposal dedup to the candidate's context
docs: document the project and each package
refactor!: reshape the API around a generic optimizer
```

Types in use: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`. Scope is the
optimizer or package (`gepa`, `opro`, `mipro`, `core`, `ai-sdk`). `!` marks a
breaking change.

The body carries the reasoning: what was wrong, what changed, what it cost. Look
at the existing log before writing one — the standard here is high and the
history is the design record.

Do not commit, branch, or push unless asked.
