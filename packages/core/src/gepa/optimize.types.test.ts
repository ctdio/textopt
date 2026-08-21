import { expect, test } from "vitest";
import { createEpochShuffledSampler } from "../sampling.js";
import type { Candidate, TextModel } from "../types.js";
import { GepaOptimizer } from "./optimize.js";
import {
  allComponentsSelector,
  fullEvaluationPolicy,
  subsampledEvaluationPolicy,
} from "./strategies.js";
import type { GepaAdapter, GepaEvent, GepaStopReason } from "./types.js";

interface Ticket {
  id: string;
  text: string;
}

const SEED = {
  systemPrompt: "Answer the ticket.",
  rubric: "Cite the policy.",
};

const TICKETS: Ticket[] = [
  { id: "a", text: "printer on fire" },
  { id: "b", text: "refund please" },
];

const REFLECT: TextModel = async () => "```\nrewritten\n```";

/**
 * Every `@ts-expect-error` below is the regression test: if a `NoInfer` is
 * deleted, the component names widen back to `string`, the error stops firing
 * and `tsc` reports the directive as unused. Nothing here is ever called — the
 * compiler is the assertion.
 */
export function componentKeysAreChecked(gepa: GepaOptimizer): unknown[] {
  return [
    gepa.optimize({
      seedCandidate: SEED,
      trainingSet: TICKETS,
      adapter: createTicketAdapter(),
      reflect: REFLECT,
      maxMetricCalls: 1,
      // @ts-expect-error "rubrik" is not a component of the seed candidate
      componentSelector: () => ["rubrik"],
    }),

    gepa.optimize({
      seedCandidate: SEED,
      trainingSet: TICKETS,
      adapter: {
        ...createTicketAdapter(),
        // @ts-expect-error "tonne" is not a component of the seed candidate
        proposeNewTexts: () => ({ tonne: "oops" }),
      },
      reflect: REFLECT,
      maxMetricCalls: 1,
    }),

    gepa.optimize({
      seedCandidate: SEED,
      trainingSet: TICKETS,
      // @ts-expect-error the adapter is keyed on components the seed lacks
      adapter: createMismatchedAdapter(),
      reflect: REFLECT,
      maxMetricCalls: 1,
    }),
  ];
}

test("infers component names from the seed candidate", async () => {
  const events: GepaEvent[] = [];
  const result = await new GepaOptimizer({
    minibatchSize: 1,
    maxIterations: 1,
    seed: 1,
  }).optimize({
    seedCandidate: SEED,
    trainingSet: TICKETS,
    adapter: createTicketAdapter(),
    reflect: REFLECT,
    maxMetricCalls: 20,
    componentSelector: allComponentsSelector(),
    batchSampler: createEpochShuffledSampler({ minibatchSize: 1 }),
    valEvaluationPolicy: fullEvaluationPolicy(),
    instanceId: ({ datum }) => datum.id,
    reporters: [{ onEvent: (event) => events.push(event) }],
  });

  result.bestCandidate.systemPrompt satisfies string;
  result.stopReason satisfies GepaStopReason;
  result.candidates[0]?.updatedComponents satisfies
    ("systemPrompt" | "rubric")[] | undefined;

  // @ts-expect-error the seed has no component called "sytemPrompt"
  const typo: unknown = result.bestCandidate.sytemPrompt;

  expect(typo).toBeUndefined();
  expect(result.bestCandidate.rubric).toBeTypeOf("string");
  expect(events[0]?.type).toBe("start");
});

test("narrows component names down to a single-component seed", async () => {
  const result = await new GepaOptimizer({
    minibatchSize: 1,
    maxIterations: 1,
    seed: 1,
  }).optimize({
    seedCandidate: { onlyOne: "Answer the ticket." },
    trainingSet: TICKETS,
    adapter: createSingleComponentAdapter(),
    reflect: REFLECT,
    maxMetricCalls: 20,
    valEvaluationPolicy: subsampledEvaluationPolicy({ size: 1 }),
  });

  // @ts-expect-error the seed declares "onlyOne" and nothing else
  const missing: unknown = result.bestCandidate.rubric;

  expect(missing).toBeUndefined();
  expect(result.bestCandidate.onlyOne).toBeTypeOf("string");
});

test("keeps the component names a widened annotation threw away", async () => {
  // The documented hole: annotating the seed as `Candidate` destroys the union
  // at the declaration, with no diagnostic anywhere. `satisfies Candidate`
  // preserves it.
  const widened: Candidate = { onlyOne: "Answer the ticket." };
  const preserved = { onlyOne: "Answer the ticket." } satisfies Candidate;

  const gepa = new GepaOptimizer({
    minibatchSize: 1,
    maxIterations: 0,
    seed: 1,
  });
  const fromWidened = await gepa.optimize({
    seedCandidate: widened,
    trainingSet: TICKETS,
    adapter: createSingleComponentAdapter(),
    reflect: REFLECT,
    maxMetricCalls: 20,
  });
  const fromPreserved = await gepa.optimize({
    seedCandidate: preserved,
    trainingSet: TICKETS,
    adapter: createSingleComponentAdapter(),
    reflect: REFLECT,
    maxMetricCalls: 20,
  });

  fromWidened.bestCandidate.anythingGoes satisfies string;
  // @ts-expect-error `satisfies` keeps the literal keys the annotation lost
  fromPreserved.bestCandidate.anythingGoes satisfies string;

  expect(fromPreserved.bestCandidate.onlyOne).toBeTypeOf("string");
});

/**
 * Feedback is what separates reflective search from blind search, and an
 * adapter that returns only scores leaves the reflection prompt printing empty
 * diagnoses — a run that looks completely normal and searches at random. The
 * shared `EvaluationBatch` leaves it optional because the optimizers that never
 * reflect have no use for it; `GepaAdapter` is where it stops being optional.
 */
export function feedbackIsRequired(): GepaAdapter<Ticket, null, string> {
  return {
    // @ts-expect-error a GEPA adapter must report per-instance feedback
    evaluate: ({ batch }) => ({
      outputs: batch.map(() => ""),
      scores: batch.map(() => 0.5),
      trajectories: batch.map(() => null),
    }),
    makeReflectiveDataset: () => ({}),
  };
}

function createTicketAdapter(): GepaAdapter<
  Ticket,
  null,
  string,
  "systemPrompt" | "rubric"
> {
  return {
    evaluate: ({ batch, candidate }) => ({
      outputs: batch.map(() => candidate.systemPrompt),
      scores: batch.map(() => 0.5),
      feedback: batch.map(() => "say more"),
      trajectories: batch.map(() => null),
    }),
    makeReflectiveDataset: ({ batch, componentsToUpdate }) =>
      Object.fromEntries(
        componentsToUpdate.map((component) => [
          component,
          batch.map((datum) => ({
            inputs: { text: datum.text },
            generatedOutputs: "",
            feedback: "say more",
          })),
        ]),
      ),
  };
}

function createMismatchedAdapter(): GepaAdapter<
  Ticket,
  null,
  string,
  "tone" | "facts"
> {
  return {
    evaluate: ({ batch }) => ({
      outputs: batch.map(() => ""),
      scores: batch.map(() => 0.5),
      feedback: batch.map(() => "say more"),
      trajectories: batch.map(() => null),
    }),
    makeReflectiveDataset: () => ({}),
  };
}

function createSingleComponentAdapter(): GepaAdapter<Ticket, null, string> {
  const adapter = createTicketAdapter();

  return {
    evaluate: ({ batch }) => ({
      outputs: batch.map(() => ""),
      scores: batch.map(() => 0.5),
      feedback: batch.map(() => "say more"),
      trajectories: batch.map(() => null),
    }),
    makeReflectiveDataset: adapter.makeReflectiveDataset,
  };
}
