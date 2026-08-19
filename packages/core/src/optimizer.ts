import type { RetryPolicy } from "./evaluation.js";
import type { Adapter, Candidate, UsageTotals } from "./types.js";

/**
 * The run-level inputs every optimizer needs, whatever search it runs. An
 * optimizer's own task type is a superset of this; the shared members are what
 * a caller can rely on without knowing which optimizer it holds.
 */
export interface OptimizerTask<
  Datum,
  Trajectory = unknown,
  Output = unknown,
  K extends string = string,
> {
  seedCandidate: Candidate<K>;
  trainingSet: readonly Datum[];
  validationSet?: readonly Datum[];
  /**
   * `NoInfer` keeps the adapter out of `K`'s inference: an adapter built by a
   * factory knows nothing about component names, and one inference candidate of
   * `string` widens `K` back to `string` everywhere.
   */
  adapter: Adapter<Datum, Trajectory, Output, NoInfer<K>>;
  /**
   * Instances held back from the search entirely, used once at the end to
   * score the winner. Selection pressure is applied to the validation set for the
   * whole run, so `bestScore` is partly fitted to it; `testScore` is the only
   * number in a result that no candidate was ever selected against.
   */
  testSet?: readonly Datum[];
  maxMetricCalls: number;
  /**
   * Dollars the run may spend, as reported by the adapter's usage. Checked
   * between evaluations rather than during one, so a run stops at the first
   * decision point past the ceiling rather than exactly on it.
   *
   * A rollout ceiling cannot bound spend on its own: reflective search grows
   * the text it optimizes, so late rollouts cost more than early ones.
   */
  maxCostUsd?: number;
  /**
   * Wall-clock milliseconds the run may take. Checked between evaluations, so
   * a run overruns by at most the length of one.
   *
   * Neither a rollout ceiling nor a cost ceiling bounds duration: a run behind
   * a rate limit spends almost nothing and takes as long as the provider
   * makes it take. This is what makes an optimizer safe to put behind a
   * request timeout or a nightly job.
   */
  maxWallClockMs?: number;
  /**
   * Names the system under optimization — model id, decoding settings, scorer
   * version — so cached scores measured under one are never served to another.
   * Change it whenever anything outside the candidate text changes.
   */
  cacheNamespace?: string;
  /**
   * How a rollout the adapter reported as an infrastructure failure is retried.
   * Optimizer-agnostic, because a rate limit costs every search the same thing:
   * an instance that measured the provider rather than the candidate.
   */
  retry?: RetryPolicy;
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
  Output = unknown,
> {
  bestCandidate: Candidate<K>;
  bestScore: number;
  bestOutputs?: (Output | undefined)[];
  metricCalls: number;
  /**
   * Tokens and dollars the run spent, summed from what the adapter reported.
   * Zero throughout when the adapter reports no usage.
   */
  usage: UsageTotals;
  /**
   * `bestCandidate`'s mean score over the held-out testSet. Absent when no
   * testSet was given. A large gap below `bestScore` is the search having
   * fitted the validation instances rather than the task.
   */
  testScore?: number;
  /**
   * Rollouts the held-out sweep cost. Reported separately because it is
   * measurement rather than search, and so is not charged to `maxMetricCalls`.
   */
  testMetricCalls?: number;
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
    Trajectory = unknown,
    Output = unknown,
    const K extends string = string,
  >(
    task: OptimizerTask<Datum, Trajectory, Output, K>,
  ): Promise<OptimizerResult<K, Stop, Output>>;
}
