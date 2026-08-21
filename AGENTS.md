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
                                     compare, deadline, checkpoint, reporting,
                                     rng, types
  src/gepa/              "./gepa"    reflective evolution over a Pareto frontier
  src/simba/             "./simba"   mini-batch ascent on program disagreement
  src/opro/              "./opro"    score-history meta-prompting
  src/mipro/             "./mipro"   joint search with a TPE surrogate
  src/bootstrap-search/  "./bootstrap-search"   few-shot search, no proposal model
  src/random-search/     "./random-search"
  src/file-cache.ts      "./file-cache"         the only entry point using node:fs
  src/testing.ts         "./testing"            fixtures shared by tests
packages/ai-sdk, packages/braintrust, packages/langchain    adapters
packages/langsmith                                          run reporting
examples/                                                   runnable scripts
bench/                                                      offline seed sweeps
docs/                                                       long-form guides
```

Each optimizer owns its directory and exports through its own entrypoint. The
substrate must stay free of anything shaped like a particular optimizer — if a
type only makes sense for reflective search, it belongs in `gepa/`.

`reporting.ts` is where that rule earns its keep. Every search emits its own
event union, but `candidateAccepted` and `finish` intersect a shared payload,
so one reporter reads a run without knowing which optimizer produced it. A new
optimizer emits `candidateAccepted` only when the incumbent moves and a full
validation sweep measured it: a row aligned with a minibatch is not a row a
reporter can name instances against.

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
edit a `version` field or write a release commit by hand.

A release is two merges. Pushing to `main` runs the release workflow, which
opens a `chore: version packages` pull request applying every pending changeset.
Merging that pull request runs the workflow again, and the second run publishes.
Both runs stop at the `npm` environment for a human approval, including the one
that only refreshes the version pull request and publishes nothing.

A push with neither a changeset waiting nor an unpublished version — a docs
commit, say — never gets that far. An environment gate stops a job before its
first step, so the release job cannot find out it has nothing to do without
first asking to be approved; a `preflight` job holding no environment answers
that question instead, and the approval is only requested for a push that will
act on it.

Both published packages are on 0.x, where `^0.1.0` does not cross a minor. So
`minor` is the breaking lever and `patch` covers additions and fixes. A changed
default that makes an unmodified call search differently is breaking. The
`/textopt-changeset` skill in `.claude/skills/textopt-changeset` applies these
rules against the branch diff.

### The registry trusts the workflow, not a secret

There is no npm token in this repository. The publish authenticates over OIDC
against a trusted publisher configured on each package, which names four things:
the user `ctdio`, the repository `textopt`, the workflow file `release.yml`, and
the environment `npm`. Rename the workflow file or the environment and
publishing stops — the registry checks both, so a rename is a break rather than
a degradation.

Two consequences worth holding onto:

`setup-node` must not set `registry-url`. It writes an `.npmrc` containing
`${NODE_AUTH_TOKEN}`, and with no such token set pnpm sends that unresolved
placeholder as the credential and takes a 404 before OIDC is tried. Fixed in
pnpm 11.1.3, but the registry it configures is already the default, so the
setting only buys back the failure mode.

Provenance needs no flag. npm attests every trusted publish by default, and a
trusted publish is the only kind it will attest — so anything published by hand
is permanently unattested.

### Promoting a package out of beta

`@textopt/ai-sdk` and `@textopt/braintrust` are `private` and listed in
`.changeset/config.json` under `ignore`, so they neither version nor publish.
Removing both markers is most of what promotes one, but not all of it.

The registry will not accept the first version from CI. A trusted publisher can
only be configured on a package that already exists, and this workflow holds no
token to bootstrap one with, so the first publish is done by hand and every
release after it runs through CI:

1. Drop `private`, and add `publishConfig: { "access": "public" }` — a scoped
   package defaults to restricted, which a free account cannot publish.
2. `pnpm build`. `files` is `["dist"]`, so publishing without a build ships an
   empty tarball and says nothing about it.
3. `pnpm publish --filter @textopt/<name>`, at version `0.0.0`. Use pnpm and not
   npm: every adapter depends on `textopt` through `workspace:*`, which npm
   publishes verbatim as an uninstallable range. Publish `0.0.0` and not the
   real version, because this one is unattested and every version anyone
   installs should have come from CI.
4. Configure the trusted publisher on npmjs.com with the four fields above.
5. Remove the package from `ignore`, and write a changeset for it.

Leave a new adapter out of the `fixed` group in `.changeset/config.json` unless
it genuinely tracks the core release for release. `fixed` holds `textopt` and
`@textopt/langchain` at one version, so anything added to it ships a new version
every time the core does, whether or not it changed.

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

The READMEs and the pages under `docs/` quote real numbers ("0.87 mean best, 8
runs in 15"). Those came from sweeps, not intuition. Two rules follow:

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

Three of the four tasks score the candidate's own text, which leaves a
demonstration search nothing to harvest; `demonstrated` models a system that is
right on some instances and wrong on others of the same kind, which is the only
condition under which harvesting is a lever at all. When adding a task, check
what a demo of it would carry: the bench datum holds the terms an answer needs,
so rendering one with the default JSON renderer prints the answer key into the
candidate and every harvesting entrant scores by reading it back. That is what
`renderBenchDemo` exists to prevent.

```bash
pnpm bench          # 4 tasks x 7 configurations x 20 seeds, a few minutes
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
