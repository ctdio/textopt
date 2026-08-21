import type { RetryPolicy } from "./evaluation.js";
import type { Adapter, Candidate, UsageTotals } from "./types.js";
import type { RunWarning } from "./warnings.js";

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
  /**
   * Instances the search draws evidence from. Reflective optimizers mine these
   * for what a candidate got wrong, so a training row earns its place by being
   * diagnostic — a row every candidate already passes teaches the rewriter
   * nothing.
   *
   * @see docs/data-prep.md
   */
  trainingSet: readonly Datum[];
  /**
   * Instances the search selects candidates against. Defaults to
   * `trainingSet`, which is the right default for a first run and the wrong
   * number to report from one — the result carries a warning saying so.
   * `"reuseTraining"` is that same default with the caller's name on it, and
   * carries no warning.
   *
   * Split by group rather than by row: near-duplicate instances that straddle
   * the boundary leak, and the run reports a score nothing earned.
   *
   * @see docs/data-prep.md
   */
  validationSet?: readonly Datum[] | "reuseTraining";
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
  /**
   * Scored rollouts the search may spend. Cache hits do not count, and test
   * rollouts are outside it entirely.
   *
   * Spending it is only worth anything if the metric separates candidates
   * first: a metric that scores every candidate alike turns the whole budget
   * into ranked ties, and the run reports a stop reason that looks like any
   * other.
   *
   * @see docs/metric-preflight.md
   * @see docs/tuning.md
   */
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
   * Rollouts served from the cache rather than charged to `metricCalls`. Every
   * optimizer here caches by default, so the same `maxMetricCalls` can buy a
   * search that revisits scored candidates a longer effective run than one
   * that never does — a comparison over `metricCalls` alone hides that. Zero
   * for a run with caching disabled.
   */
  cacheHits: number;
  /**
   * Calls made to a proposal or reflection model, which no metric budget
   * covers — see each optimizer's own accounting for what a call costs there.
   * Absent from a search that proposes nothing of its own: bootstrap search
   * only accepts or rejects rollouts the metric already scored.
   */
  reflectionCalls?: number;
  /**
   * Tokens and dollars the search spent, summed from what the adapter reported.
   * Zero throughout when the adapter reports no usage. `maxCostUsd` is checked
   * against this, so the held-out sweep — which runs after the search has
   * stopped, under no ceiling — is reported apart from it, in `testUsage`.
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
  /**
   * Tokens and dollars the held-out sweep cost, for the same reason: no ceiling
   * bounds it, so a caller adding up what a run spent has to see it as its own
   * number rather than find it folded into one `maxCostUsd` was supposed to
   * hold. Absent when no testSet was given.
   */
  testUsage?: UsageTotals;
  stopReason: Stop;
  /**
   * What this run cannot say about itself from its own numbers — selection
   * that reused the training instances, a seed the metric could not separate.
   * Empty when there is nothing to say. Never fatal, and repeated on the
   * `finish` event so a reporter sees them next to the score.
   */
  warnings: RunWarning[];
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
