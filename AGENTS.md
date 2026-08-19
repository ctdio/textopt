# Working in this repository

textopt is a TypeScript library for optimizing the text inside an LLM system —
prompts, instructions, few-shot blocks. It implements four published search
algorithms over a shared substrate, plus adapters for common frameworks.

Read this before changing anything. The conventions below are not stylistic
preferences; most of them exist because breaking them has already cost someone
a day.

## Layout

```
packages/core        the substrate and every optimizer
  src/               "."            evaluation, budget, cache, demos, rng, types
  src/gepa/          "./gepa"       reflective evolution over a Pareto frontier
  src/opro/          "./opro"       score-history meta-prompting
  src/mipro/         "./mipro"      joint search with a TPE surrogate
  src/random-search/ "./random-search"
  src/testing.ts     "./testing"    fixtures shared by tests
packages/ai-sdk, packages/braintrust, packages/langchain    adapters
examples/                                                   runnable scripts
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
npx prettier --write <paths>                # before committing
```

Never re-run a suite just to re-read a failure. The output is already in the
log file.

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

Use a scratch script under the session scratchpad for sweeps, and delete it
afterwards. Sweeps do not belong in the test suite; a focused regression test
pinned to one seed does.

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
