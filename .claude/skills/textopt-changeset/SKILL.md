---
name: textopt-changeset
description: Draft a changeset for the current branch's changes — work out which packages are affected, pick the bump level under 0.x semver, and write .changeset/<slug>.md in this repository's voice. Use when the user says "/textopt-changeset", "write a changeset", "add a release note", "what bump does this need", or has finished a change that ships to npm.
---

# Changeset

Changesets records release intent as a file, not as a version bump. This skill
writes that file. `changeset version` later consumes it, computes the numbers,
and writes each package's `CHANGELOG.md`. See `## Releasing` in `AGENTS.md`.

## Decide whether one is needed at all

Read the change before classifying it. A changeset describes what a caller of a
published package can now do differently — not what moved in the source tree.

| Changed                                              | Package                                                                                                                           |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `packages/core/**`                                   | `textopt`                                                                                                                         |
| `packages/langchain/**`                              | `@textopt/langchain`                                                                                                              |
| `packages/langsmith/**`                              | `@textopt/langsmith`                                                                                                              |
| `packages/ai-sdk/**`, `packages/braintrust/**`       | none — both are `private` and listed under `ignore` in `.changeset/config.json`, so a changeset naming them is silently discarded |
| `examples/**`, `bench/**`, `.github/**`, root config | none                                                                                                                              |

Within a published package, these do not earn a changeset on their own:
`*.test.ts`, `tsconfig.json`, `tsdown.config.ts`. Nothing a consumer installs
changes.

`packages/core` lists `docs` in its `files` alongside `dist`, so
`packages/core/docs/**` and any package `README.md`
all ship in the tarball and are things a consumer installs. A change to one
takes a `patch` if it is worth releasing and nothing otherwise. Guidance that
contradicts the code it ships beside is worth releasing.

If the only affected packages are ignored ones, say so and stop. Do not write a
file that changesets will throw away.

## Pick the bump

Both published packages are on 0.x, where npm resolves `^0.1.0` without
crossing a minor. That inverts the usual meaning:

- **`minor`** — the breaking lever. An export removed or renamed, a signature
  or required option changed, or a **default changed such that an unmodified
  call now searches differently**. That last one is the easiest to miss and the
  one this codebase cares most about; see `## Fidelity to the papers`.
- **`patch`** — everything else. New exports, new optional options, bug fixes,
  performance, corrected behaviour that was plainly wrong.
- **`major`** — not during 0.x. It is reserved for the deliberate 1.0.

When one branch carries several unrelated changes, write several files. They
aggregate into one release but stay separate changelog lines, which is what a
reader wants.

## Write it

Name the file for the change, not randomly: `.changeset/mipro-multivariate-default.md`.

```md
---
"textopt": minor
---

MIPRO's TPE surrogate models complete configurations by default. Set
`multivariate: false` to model each component independently.
```

List only the packages whose changelog should carry this text. `textopt` and
`@textopt/langchain` are a `fixed` pair, so both take the version bump either
way — naming a package decides where the prose lands, not who gets bumped.

The prose follows the repository's voice, as in `## Commits` in `AGENTS.md`:
present tense, declarative, no "we" and no "this PR". Name the actual API.
State the behaviour, then the migration if there is one. One or two sentences.
It is read by someone deciding whether to upgrade, not by a reviewer who
already knows the diff.

## Validate

```bash
pnpm changeset status
```

Confirm the packages listed are the ones you intended and the bump matches.
Report the resulting version to the user — `status` prints the bump level, and
the current versions are in each `package.json`.
