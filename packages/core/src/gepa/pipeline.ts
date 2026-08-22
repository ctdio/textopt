import { mapWithConcurrency } from "../concurrency.js";
import type {
  GepaAdapter,
  ReflectiveBatch,
  ReflectiveRecord,
} from "./types.js";
import type { Candidate, EvaluationBatch, ScoreResult } from "../types.js";

/** One module of the system: a component's text, applied to what reached it. */
export interface PipelineModule<Datum, K extends string> {
  /** The candidate component holding this module's instruction. */
  component: K;
  run: (args: {
    instruction: string;
    /** The previous module's output, or the pipeline input for the first. */
    input: unknown;
    datum: Datum;
    signal?: AbortSignal;
  }) => Promise<unknown> | unknown;
}

export interface PipelineStep {
  component: string;
  input: unknown;
  output: unknown;
}

/** What each module in one rollout received and produced, in order. */
export interface PipelineTrace {
  steps: PipelineStep[];
}

const DEFAULT_CONCURRENCY = 1;

/**
 * A GEPA adapter for a system built from several modules in sequence, where
 * each module's instruction is its own candidate component.
 *
 * The work this saves is attribution. Reflection on a multi-module system is
 * only as good as the evidence it sees, and the evidence a module needs is what
 * *it* received and produced — not the pipeline's input and final answer, which
 * is what an adapter written in a hurry ends up showing every component.
 *
 * The feedback is end-to-end, and every module sees the same string. That is
 * the honest default rather than a shortcut: a metric scores the final output,
 * so nothing in a score alone says which module lost the point. A caller who
 * can attribute better should score the steps themselves — `score` is handed
 * the whole trace for exactly that.
 *
 * Errors from a module are not caught. A helper cannot tell a rate limit from a
 * bug in a module, and guessing wrong either buries the bug or fails the run
 * over a blip. Classify inside `run` and return a transient `ScoreResult` from
 * `score`, or let the optimizer's `raiseOnError` decide.
 */
export function createPipelineAdapter<
  Datum,
  Output = unknown,
  K extends string = string,
>(args: {
  modules: readonly PipelineModule<Datum, K>[];
  /** What the first module receives. Defaults to the datum itself. */
  input?: (datum: Datum) => unknown;
  score: (args: {
    datum: Datum;
    output: Output;
    steps: readonly PipelineStep[];
  }) => Promise<ScoreResult> | ScoreResult;
  /** Rollouts in flight at once. Default 1. */
  concurrency?: number;
}): GepaAdapter<Datum, PipelineTrace, Output, K> {
  const {
    modules,
    input = (datum: Datum) => datum,
    score,
    concurrency = DEFAULT_CONCURRENCY,
  } = args;

  if (modules.length === 0) {
    throw new Error("createPipelineAdapter requires at least one module");
  }

  return {
    evaluate: async ({
      batch,
      candidate,
      captureTraces,
      onRollout,
      signal,
    }) => {
      const rollouts = await mapWithConcurrency({
        items: batch,
        limit: concurrency,
        ...(onRollout === undefined ? {} : { onSettled: onRollout }),
        signal,
        task: async (datum) => {
          const trace = await runPipeline({
            modules,
            candidate,
            datum,
            input,
            signal,
          });
          const output = (trace.steps.at(-1) as PipelineStep).output as Output;

          return {
            trace,
            output,
            scored: await score({ datum, output, steps: trace.steps }),
          };
        },
      });

      const batchResult: ReflectiveBatch<PipelineTrace, Output> = {
        outputs: rollouts.map((rollout) => rollout.output),
        scores: rollouts.map((rollout) => rollout.scored.score),
        feedback: rollouts.map((rollout) => rollout.scored.feedback ?? ""),
        usage: rollouts.map((rollout) => rollout.scored.usage ?? {}),
        transient: rollouts.map((rollout) => rollout.scored.transient ?? false),
        objectiveScores: rollouts.map(
          (rollout) => rollout.scored.objectiveScores ?? {},
        ),
      };

      return captureTraces
        ? { ...batchResult, trajectories: rollouts.map((r) => r.trace) }
        : batchResult;
    },

    makeReflectiveDataset: ({ evaluation, componentsToUpdate }) => {
      const dataset: Partial<Record<K, ReflectiveRecord[]>> = {};

      for (const component of componentsToUpdate) {
        dataset[component] = recordsFor({ component, evaluation });
      }

      return dataset;
    },
  };
}

async function runPipeline<Datum, K extends string>(args: {
  modules: readonly PipelineModule<Datum, K>[];
  candidate: Candidate<K>;
  datum: Datum;
  input: (datum: Datum) => unknown;
  signal?: AbortSignal;
}): Promise<PipelineTrace> {
  const { modules, candidate, datum, input, signal } = args;
  const steps: PipelineStep[] = [];
  let carried = input(datum);

  for (const module of modules) {
    const output = await module.run({
      instruction: candidate[module.component],
      input: carried,
      datum,
      signal,
    });
    steps.push({ component: module.component, input: carried, output });
    carried = output;
  }

  return { steps };
}

function recordsFor<Output>(args: {
  component: string;
  evaluation: EvaluationBatch<PipelineTrace, Output>;
}): ReflectiveRecord[] {
  const { component, evaluation } = args;
  const traces = evaluation.trajectories ?? [];

  return traces.map((trace, index) => {
    const step = trace.steps.find((entry) => entry.component === component);

    return {
      inputs: step?.input,
      generatedOutputs: step?.output,
      feedback: evaluation.feedback?.[index] ?? "",
      score: evaluation.scores[index],
      evidence: { component },
    };
  });
}
