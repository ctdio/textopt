import { createPipelineAdapter } from "./pipeline.js";
import type { PipelineTrace } from "./pipeline.js";
import type { GepaAdapter } from "./types.js";
import type { Candidate, ScoreResult } from "../types.js";

/**
 * A GEPA adapter for the common case: one prompt, one component, one call.
 *
 * `run` receives the candidate's text as `instruction` and returns whatever the
 * system produced; `score` grades that output. The reflective dataset then
 * carries what the prompt received and what it produced, which is all the
 * evidence there is when the system is a single call.
 *
 * The candidate must have exactly one component, and that is the point rather
 * than a limitation of the helper. A second component would be text the search
 * rewrites every iteration and nothing ever runs — a budget spent on proposals
 * that cannot move the score, reported as an ordinary run. A system with
 * several instructions wants `createPipelineAdapter`, which gives each one a
 * module and each module its own evidence.
 */
export function createPromptAdapter<
  Datum,
  Output = unknown,
  K extends string = string,
>(args: {
  run: (args: {
    instruction: string;
    /** What the prompt received: `input(datum)`, or the datum itself. */
    input: unknown;
    datum: Datum;
    signal?: AbortSignal;
  }) => Promise<Output> | Output;
  /** What the prompt receives. Defaults to the datum itself. */
  input?: (datum: Datum) => unknown;
  score: (args: {
    datum: Datum;
    output: Output;
  }) => Promise<ScoreResult> | ScoreResult;
  /** Rollouts in flight at once. Default 1. */
  concurrency?: number;
}): GepaAdapter<Datum, PipelineTrace, Output, K> {
  const { run, input, score, concurrency } = args;

  // The component is read off the candidate rather than named up front, so a
  // caller who later adds a second one is told instead of silently optimizing
  // text nothing runs. Building the delegate per call keeps that check honest;
  // it allocates an object literal beside a batch of rollouts.
  function delegateFor(
    candidate: Candidate<K>,
  ): GepaAdapter<Datum, PipelineTrace, Output, K> {
    return createPipelineAdapter<Datum, Output, K>({
      modules: [{ component: soleComponentOf(candidate), run }],
      ...(input === undefined ? {} : { input }),
      ...(concurrency === undefined ? {} : { concurrency }),
      score,
    });
  }

  return {
    evaluate: async (evaluateArgs) =>
      delegateFor(evaluateArgs.candidate).evaluate(evaluateArgs),

    makeReflectiveDataset: (datasetArgs) =>
      delegateFor(datasetArgs.candidate).makeReflectiveDataset(datasetArgs),
  };
}

function soleComponentOf<K extends string>(candidate: Candidate<K>): K {
  const components = Object.keys(candidate) as K[];

  if (components.length === 1) {
    return components[0] as K;
  }

  throw new Error(
    `createPromptAdapter requires a candidate with exactly one component, and this one has ${components.length}` +
      `${components.length === 0 ? "" : ` (${components.join(", ")})`}. ` +
      "Use createPipelineAdapter to give each instruction a module that runs it.",
  );
}
