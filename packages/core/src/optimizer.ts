import type { Adapter, Candidate } from "./types.js";

/**
 * The run-level inputs every optimizer needs, whatever search it runs. An
 * optimizer's own task type is a superset of this; the shared members are what
 * a caller can rely on without knowing which optimizer it holds.
 */
export interface OptimizerTask<
  Datum,
  Traj = unknown,
  Out = unknown,
  K extends string = string,
> {
  seedCandidate: Candidate<K>;
  trainset: readonly Datum[];
  valset?: readonly Datum[];
  /**
   * `NoInfer` keeps the adapter out of `K`'s inference: an adapter built by a
   * factory knows nothing about component names, and one inference candidate of
   * `string` widens `K` back to `string` everywhere.
   */
  adapter: Adapter<Datum, Traj, Out, NoInfer<K>>;
  maxMetricCalls: number;
  signal?: AbortSignal;
}

/**
 * What every optimizer reports. `Stop` has no default: an optimizer that cannot
 * enumerate the reasons it stops has not finished being designed, and a default
 * here would put `string` back.
 */
export interface OptimizerResult<
  K extends string,
  Stop extends string,
  Out = unknown,
> {
  bestCandidate: Candidate<K>;
  bestScore: number;
  bestOutputs?: (Out | undefined)[];
  metricCalls: number;
  stopReason: Stop;
}

/**
 * An optimizer: a task in, the best candidate it found out. Exactly one method.
 *
 * `Datum` and `K` are inferred per call, so they live on the method rather than
 * on the interface — a class parameter could only be fixed at `new`, where no
 * seed candidate exists yet.
 *
 * The interface names the contract; it is not a type to hold instances in.
 * Method parameters are bivariant, which is what lets an optimizer with a
 * richer task type implement it at all, and equally what lets a task missing
 * that optimizer's own inputs typecheck against this signature.
 */
export interface Optimizer<Stop extends string> {
  optimize<
    Datum,
    Traj = unknown,
    Out = unknown,
    const K extends string = string,
  >(
    task: OptimizerTask<Datum, Traj, Out, K>,
  ): Promise<OptimizerResult<K, Stop, Out>>;
}
