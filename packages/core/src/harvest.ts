import { createBudget } from "./budget.js";
import { createEvaluator } from "./evaluation.js";
import type { Rng } from "./rng.js";
import type { Adapter, Candidate } from "./types.js";

/**
 * One rollout worth keeping: what went in, what the system produced, and how
 * that output scored. Harvested, never written — the value of the pair is that
 * the system actually produced the output and the metric actually rewarded it.
 */
export interface Rollout<Datum = unknown, Output = unknown> {
  input: Datum;
  output: Output;
  score: number;
}

export interface HarvestResult<Datum, Output> {
  rollouts: Rollout<Datum, Output>[];
  /** Rollouts this cost. Harvesting is cheap, not free. */
  metricCalls: number;
  /** Instances run, including the ones the metric did not reward. */
  attempted: number;
}

/**
 * Run a candidate over data and keep the rollouts the metric rewarded.
 *
 * The library's one paid collection primitive, with two consumers: a few-shot
 * block wants four of these, and a distillation set wants thousands. Both are
 * the same pass — run the candidate, score it, keep what cleared the bar — so
 * both share the budget, retry and transient-failure handling that pass needs.
 *
 * Which data to sweep is the caller's decision and the consequential one. A
 * validation set is the wrong choice: it is the set that selected the candidate,
 * so the rollouts it yields are enriched for the candidate's fit to those
 * instances rather than to the task. Prefer the training set, or a pool held
 * out of the run entirely.
 */
export async function harvestRollouts<
  Datum,
  Trajectory,
  Output,
  K extends string = string,
>(args: {
  adapter: Adapter<Datum, Trajectory, Output, K>;
  /** The candidate to run. Usually a run's winner, sometimes the seed. */
  candidate: Candidate<K>;
  data: readonly Datum[];
  /**
   * Score a rollout must reach to be kept. Unset keeps every rollout the
   * metric rewarded at all, which is what MIPROv2's bootstrapper does without
   * a `metric_threshold`: it keeps a trace on any truthy score and only
   * compares against a number once one is configured.
   *
   * Demanding a perfect score instead is the right call for a boolean metric
   * and the wrong one for a graded metric, where it throws away every rollout
   * that was most of the way there — which on a hard task is all of them.
   */
  minScore?: number;
  /** Rollouts to collect before stopping. Unset sweeps the whole pool. */
  maxRollouts?: number;
  /**
   * Instances per rollout batch. Smaller batches stop closer to the moment
   * enough rollouts exist, at the cost of less concurrency inside the adapter.
   */
  batchSize?: number;
  /** Ceiling on rollouts run. Defaults to one pass over `data`. */
  maxMetricCalls?: number;
  /** Shuffles `data` first, so rollouts are not all drawn from its head. */
  rng?: Rng;
  signal?: AbortSignal;
}): Promise<HarvestResult<Datum, Output>> {
  const {
    adapter,
    candidate,
    data,
    minScore,
    maxRollouts = Number.POSITIVE_INFINITY,
    batchSize = Math.min(maxRollouts, data.length),
    maxMetricCalls = data.length,
    rng,
    signal,
  } = args;

  if (data.length === 0) {
    throw new Error("harvestRollouts requires non-empty data");
  }

  const budget = createBudget({ maxMetricCalls });
  // Uncached on purpose: the cache stores scores, and a rollout is kept for the
  // output it produced, which a cache hit cannot return.
  const evaluator = createEvaluator<Datum, Trajectory, Output, K>({
    adapter,
    budget,
    ...(signal === undefined ? {} : { signal }),
  });

  const order = rng === undefined ? [...data] : rng.shuffle(data);
  const rollouts: Rollout<Datum, Output>[] = [];
  let attempted = 0;

  for (let start = 0; start < order.length; start += batchSize) {
    if (rollouts.length >= maxRollouts || signal?.aborted) {
      break;
    }

    const batch = order.slice(
      start,
      start + Math.min(batchSize, budget.remaining()),
    );
    if (batch.length === 0) {
      break;
    }

    const evaluation = await evaluator.evaluateTraced({
      candidate,
      batch,
      split: "train",
      phase: "seed",
      candidateId: null,
      iteration: 0,
    });
    if (evaluation === null) {
      break;
    }
    attempted += batch.length;

    for (let index = 0; index < batch.length; index += 1) {
      const score = evaluation.scores[index] as number;
      const rewarded = minScore === undefined ? score > 0 : score >= minScore;
      if (!rewarded || rollouts.length >= maxRollouts) {
        continue;
      }
      rollouts.push({
        input: batch[index] as Datum,
        output: evaluation.outputs[index] as Output,
        score,
      });
    }
  }

  return { rollouts, metricCalls: budget.spent(), attempted };
}
