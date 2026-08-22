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
  docs/                                    long-form guides
  evals/                                   eval cases measuring those guides
packages/ai-sdk, packages/braintrust, packages/langchain    adapters
packages/langsmith                                          run reporting
examples/                                                   runnable scripts
bench/                                                      offline seed sweeps
```

`docs/` sits inside the package rather than at the repository root because it is
listed in its `files`. A guide that ships with the code it
describes cannot document an API the installed version does not have, which is
the whole reason it moved — and it means a change to it is a change to what a
consumer installs. See `## Releasing`.

Each optimizer owns its directory and exports through its own entrypoint. The
substrate must stay free of anything shaped like a particular optimizer — if a
type only makes sense for reflective search, it belongs in `gepa/`.

Two shared seams every optimizer goes through, whatever it searches.
`resolveValidationSet` from `warnings.ts` is how a task's `validationSet`
becomes the instances a run selects against: it is the only place that knows
`"reuseTraining"` and the only thing that reports the reuse. And every result
and `finish` event carries `warnings` — what a run could see about its own
measurement that its numbers cannot say. A new optimizer wires both; a search
that resolves its own validation set silently reintroduces the footgun.

`reporting.ts` is where that rule earns its keep. Every search emits its own
event union, but five members — `start`, `evaluation`, `rollout`,
`candidateAccepted` and `finish` — intersect shared payloads, so one reporter
reads a run without knowing which optimizer produced it. A new optimizer emits
all five, names its events in its own `<NAME>_EVENT_TYPES` list, and emits
`candidateAccepted` only when the incumbent moves and a full validation sweep
measured it: a row aligned with a minibatch is not a row a reporter can name
instances against.

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

## A feature lands in every search, or it is not a feature

Six algorithms over one substrate is the whole claim, so anything the substrate
gains is owed by all six. The failure mode is not writing wrong code — it is
writing correct code for the one search you happened to have open.

**Wire it into all six, and make the compiler hold it there.** Adding `rollout`
to the event substrate was six unions, six `<NAME>_EVENT_TYPES` lists, six
`emit` calls, and the adapters that do the counting. Those lists exist so the
work cannot be left half-done: each is `as const satisfies readonly
XEvent["type"][]`, `reporting.types.test.ts` asserts the reverse assignment, and
both stop compiling the moment a union and its list disagree.

**A check that fires on correct code is worse than no check.** The first version
of the reporter check compared handler names against the running optimizer's
list alone, so a reporter written once and attached to three searches warned on
every run about a handler doing exactly what its author meant. Before adding a
warning, name the legitimate setups that trip it, starting with the ones this
library advertises. A warning the reader learns to skip costs more than the
silence it replaced, because it stands next to warnings that matter.

**Check what a helper requires before filing it under one optimizer.**
`createPromptAdapter` lives in `gepa/` and returns a `GepaAdapter`, which is the
base `Adapter` with reflection's evidence added — so the other five take it
unchanged. Nothing prevented that; the docs merely implied otherwise, and a
SIMBA user reading them writes an adapter by hand.

**A capability with no name is undiscovered.** Optimizing a single prompt was a
`modules` array of length one for as long as `createPipelineAdapter` has
existed. An integrator on neither framework adapter wrote `evaluate` and
`makeReflectiveDataset` themselves rather than find it, because nothing was
named for the case they had. Expressible is not discoverable: when the common
case takes a sentence to explain, it wants an export.

**A claim about types is asserted, the way a claim about numbers is measured.**
`everyEventNameIsListed`, `oneReporterFitsEveryOptimizer` and
`promptAdapterFitsEveryOptimizer` are never called — each is an assignment the
compiler accepts or rejects, and each holds a sentence of documentation that
would otherwise quietly stop being true. Five lines beats a paragraph nobody
rechecks. `## Claims in documentation are measured` covers the numbers.

## Claims in documentation are measured

The READMEs and the pages under `packages/core/docs/` quote real numbers ("0.87 mean best, 8
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

## The shipped docs are evaluated

`docs/` ships in the tarball and its consumer is often an agent, so it gets
measured: given the docs and a task with a trap in it, does an agent reach the
right answer? Cases live in `packages/core/evals/`, which is not in the
package's `files` and never ships.

```bash
# early access, enabled per organization; the variable is for machines that
# cannot receive the rollout. Set it in the shell or ~/.claude/settings.json
# `env` — a repo's .claude/settings.json cannot grant it, because project
# settings only apply allowlisted variables and this is not one.
export CLAUDE_CODE_WALNUT_SPIRE=1

claude plugin eval --eval-dir packages/core/evals --ablation none \
  --scaffold --allow-tools Bash Write Edit \
  --output-dir packages/core/evals/results
```

**There is no ablation, because there is nothing to ablate.** An earlier version
of this suite shipped an agent skill and measured it with `--ablation
with-without`. Seven cases returned Δ 0.00, every one of them, including the two
hardest: a frontier model with `docs/` on disk funds a SIMBA run past its
finalist reserve and refuses a saturated metric just as reliably without the
skill as with it. The skill was dropped and its net-new content became
`docs/data-prep.md` and `docs/metric-preflight.md`. What survives is a docs
gate: if a guide is rewritten badly, the case that depends on it fails.

Δ is a tempting instrument and a bad one here. A grader that names a library
identifier is one a no-docs arm structurally cannot pass, so it scores whether
the docs were readable; a grader general enough for that arm to attempt is one a
frontier model already passes unaided. The number to read is the pass rate.

An executable case is a directory with `case.yaml` naming a `scaffold_script`
**path** (inline bash is read as a filename and fails), `prompt.md`, a
`fixtures/` directory, and graders that read the oracle's verdict:

```
splits-a-leaky-dataset/
  case.yaml            context.scaffold_script: fixtures/scaffold.sh
  prompt.md            the task, and the output contract it must satisfy
  fixtures/scaffold.sh builds the workspace
  fixtures/make-data.mjs   deterministic dataset, with the trap in it
  fixtures/verify.mjs      the oracle: re-derives every judgment, writes a verdict
  graders/*.md         regex over { source: file, path: out/verdict.json }
```

The scaffold copies `dist/` and `docs/` into `node_modules/textopt` rather than
installing a tarball — no network, and a full GEPA run against
`textopt/testing`'s offline stand-ins finishes in milliseconds, so an oracle can
afford to re-run an agent's own configuration to check it.

| Case                          | What it pins                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| `splits-a-leaky-dataset`      | Near-duplicate families are grouped before splitting, not after                      |
| `builds-a-metric-that-gates`  | A non-negotiable requirement zeroes the score; a watch-only one moves it not at all  |
| `funds-the-simba-reserve`     | The run is funded past SIMBA's validation reserve without brute-forcing the budget   |
| `refuses-a-saturated-run`     | A seed already at the ceiling stops the run instead of spending the budget on ties   |
| `enforces-a-requirement-once` | A gated requirement is not also weighted heavily, which would eat the metric's range |

A case that only asserts a property of this library belongs in a unit test
instead, where it runs free on every commit. What earns a place here is a
judgment an agent has to make from the docs.

Two rules, both learned by breaking them:

**Grade outcomes, never mechanisms.** A rubric here failed two agents for
computing a dashboard-only criterion outside the judge instead of using
`weight: 0` — the mechanism it named in its fail clause was one its own pass
clause allowed, and the excluded-by-construction design was the better answer
anyway, since `weight: 0` still leaves a criterion in `objectiveScores` where an
objective frontier selects on it. A split verdict from an LLM grader
(`FAIL PASS FAIL` on one run, `PASS FAIL PASS` on the next) is a verdict on the
grader's validity, not noise to average away.

**Pre-flight a case before trusting it**, exactly as `metric-preflight.md` tells
a user to pre-flight theirs. Write a known-bad and a known-good solution, run
both through the oracle, and confirm the checks separate them. A case whose
correct answer cannot score full marks is a broken case, and you will not learn
that from a run.

A case listing `Bash` in `allowed_tools` still does not get it: gated tools
need the operator grant `--allow-tools Bash Write Edit` on the command line as
well. Omit it and every executable case scores 0.00 in both arms with
`Bash called 0x` — a broken invocation that reads exactly like a broken case.

Runs cost real money — a prose case is roughly $0.30 an agent run and an
executable one more, so this is a run-on-docs-change gate, not CI. Watch for
`max_turns` exhaustion: a case that runs out of turns produces no final message,
and any grader reading `last_message` then scores nothing.

Results land in `packages/core/evals/results/` and are gitignored. A umask of
002 makes that directory group-writable, which the harness refuses to write
into — `chmod g-w` it. With no plugin target the harness resolves the
location to `packages/` rather than the eval dir, so pass `--output-dir`
as well.

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
- **The compiler is a test runner.** A `*.types.test.ts` file holds what no
  runtime assertion can reach — an event union against its name list, a helper's
  adapter against the interface every optimizer takes. Nothing in them is
  called; a failing `pnpm typecheck` is the red.

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
