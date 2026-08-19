# @textopt/langsmith

LangSmith experiment reporting for [textopt](https://github.com/ctdio/textopt#readme).

`createLangSmithReporter` writes a GEPA run to LangSmith as **one experiment per accepted candidate over one fixed dataset**. Selecting several of those experiments in LangSmith's comparison view renders the candidate x instance score matrix Pareto selection reads: which instances a candidate won, and which it paid for.

Install `langsmith` separately. This package matches its `Client` structurally and declares no runtime dependency on it.

## Usage

```ts
import { GepaOptimizer } from "textopt/gepa";
import { createLangSmithReporter } from "@textopt/langsmith";
import { Client } from "langsmith";

const instanceId = ({ datum }: { datum: Ticket }) => datum.id;

const result = await new GepaOptimizer({ trackBestOutputs: true }).optimize({
  seedCandidate,
  trainingSet,
  validationSet,
  testSet,
  adapter,
  reflect,
  maxMetricCalls: 300,
  instanceId,
  reporters: [
    createLangSmithReporter({
      client: new Client(),
      dataset: "ticket-triage",
      experimentPrefix: "gepa-2026-08-19",
      validationSet,
      testSet,
      instanceId,
      toInput: (datum) => ({ ticket: datum.text }),
      toExpected: (datum) => ({ label: datum.label }),
    }),
  ],
});
```

Experiments are named `<experimentPrefix>/cand-<candidateId>`, starting with `cand-0` for the seed — the baseline every later candidate is read against. Each carries the candidate's text, iteration, parents, source and aggregate score as metadata, so a score that moved sits next to the edit that moved it.

| Option             | Default              | Effect                                                              |
| ------------------ | -------------------- | ------------------------------------------------------------------- |
| `client`           | required             | A LangSmith `Client`, or anything with the same shape.              |
| `dataset`          | required             | Dataset holding the validation split. Created once, then reused.    |
| `experimentPrefix` | required             | Names the experiments. Use a value unique to the run.               |
| `validationSet`    | required             | The validation set the run was given, in the same order.            |
| `testSet`          | none                 | Uploaded as its own dataset, swept once, for the winner only.       |
| `testDataset`      | `<dataset>-held-out` | Names that second dataset.                                          |
| `instanceId`       | the row's position   | Keys a dataset row across runs.                                     |
| `toInput`          | the datum itself     | The example's `inputs`.                                             |
| `toExpected`       | none                 | The example's `outputs`, when the dataset should carry a reference. |
| `concurrency`      | `8`                  | Rows uploaded at once within one experiment.                        |

## What is deliberately not logged

**Minibatch rollouts.** They are small random subsets of the _training_ set that differ every iteration, so they cannot be compared across candidates and would bury the experiments that can. They are still traceable: with `LANGSMITH_TRACING=1`, `@textopt/langsmith`'s sibling `@textopt/langchain` tags every rollout with its iteration, phase, split and candidate id.

**A held-out experiment per candidate.** GEPA scores the held-out set only once selection is over, precisely so that nothing can be chosen against it. There is one held-out experiment, for the winner, on its own dataset — and no option to change that, because the first time a candidate is picked because it looked better there, the number stops meaning anything.

**Zeros for unmeasured instances.** An instance the evaluation policy skipped, or one an infrastructure failure lost, has no row. Written as a zero it reads as a regression that never happened.

## Notes

Uploads are queued, not awaited: `onEvent` runs on the search's hot path, and the reporter's `flush` is awaited once as the run ends, including when it ends by throwing. A LangSmith that is unreachable degrades to a warning — the run that bought the rollouts finishes either way.

Dataset row ids are derived from the dataset name and `instanceId`, not assigned by the server, so a repeated or resumed run writes to the rows it already has. If the dataset already exists, its examples are left alone; a validation set that changed shape wants a new `dataset` name.

Uploading is chatty: one run plus one feedback per measured instance per accepted candidate. Ten candidates over a hundred validation instances is a few thousand calls.
