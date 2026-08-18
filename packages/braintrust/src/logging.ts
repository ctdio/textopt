import type { Adapter, EvaluateArgs } from "textopt";

export interface BraintrustEvent {
  input?: unknown;
  output?: unknown;
  expected?: unknown;
  scores?: Record<string, number>;
  metadata?: Record<string, unknown>;
}

/**
 * Structural type satisfied by both a braintrust `Experiment` (from `init`)
 * and a `Logger` (from `initLogger`).
 */
export interface BraintrustLoggerLike {
  log(event: BraintrustEvent): unknown;
}

export interface BraintrustLoggingOptions<
  Datum,
  Traj,
  Out,
  K extends string,
  Wrapped,
> {
  /**
   * `Wrapped` carries whatever the adapter has beyond `evaluate` — a GEPA
   * adapter's `makeReflectiveDataset`, say — so the decorated adapter is still
   * accepted everywhere the undecorated one was. `K` is inferred from the
   * adapter's own component names, so decorating one does not widen them back
   * to `string`.
   */
  adapter: Wrapped & Adapter<Datum, Traj, Out, K>;
  logger: BraintrustLoggerLike;
  metadata?: Record<string, unknown>;
  toInput?: (datum: Datum) => unknown;
  toExpected?: (datum: Datum) => unknown;
}

/**
 * Wraps any adapter so every rollout lands in Braintrust. Because it decorates
 * the adapter rather than replacing it, this composes with the LangChain and AI
 * SDK adapters — or with an adapter you wrote yourself.
 *
 * Logging never fails a rollout: an unreachable logger degrades to a warning.
 */
export function withBraintrustLogging<
  Datum,
  Traj,
  Out,
  K extends string,
  Wrapped,
>(
  options: BraintrustLoggingOptions<Datum, Traj, Out, K, Wrapped>,
): Wrapped & Adapter<Datum, Traj, Out, K> {
  const { adapter, logger, metadata, toInput, toExpected } = options;

  return {
    ...adapter,

    evaluate: async (args: EvaluateArgs<Datum, K>) => {
      const evaluation = await adapter.evaluate(args);

      args.batch.forEach((datum, index) => {
        const event: BraintrustEvent = {
          input: toInput === undefined ? datum : toInput(datum),
          output: evaluation.outputs[index],
          scores: {
            score: evaluation.scores[index] ?? 0,
            ...evaluation.objectiveScores?.[index],
          },
          metadata: {
            ...metadata,
            candidate: args.candidate,
            feedback: evaluation.feedback?.[index] ?? "",
            // Group and filter a run's events by where they came from: without
            // these an experiment is one undifferentiated pile of rollouts.
            iteration: args.run.iteration,
            phase: args.run.phase,
            split: args.run.split,
            candidateId: args.run.candidateId,
          },
        };

        if (toExpected !== undefined) {
          event.expected = toExpected(datum);
        }

        try {
          logger.log(event);
        } catch (err) {
          console.warn("[textopt-braintrust] failed to log evaluation", {
            err,
          });
        }
      });

      return evaluation;
    },
  };
}
