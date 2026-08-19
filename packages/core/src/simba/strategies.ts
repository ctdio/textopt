import type { Rng } from "../rng.js";

/** One sample of one instance: which program produced it, and how it scored. */
export interface SimbaRollout<Output = unknown> {
  programIndex: number;
  score: number;
  output?: Output;
  feedback?: string;
}

/**
 * Every sample of one training instance, sorted best first, with the spread
 * that decides whether the instance is worth learning from.
 */
export interface SimbaBucket<Datum = unknown, Output = unknown> {
  index: number;
  datum: Datum;
  rollouts: SimbaRollout<Output>[];
  maxToMinGap: number;
  maxScore: number;
  maxToAvgGap: number;
}

export interface SimbaSample<Output = unknown> {
  programIndex: number;
  scores: readonly number[];
  outputs?: readonly (Output | undefined)[];
  feedback?: readonly string[];
}

/**
 * Group the step's samples by instance and rank the instances by how much the
 * programs disagreed about them.
 *
 * Disagreement is the signal SIMBA runs on. An instance every program gets
 * right teaches nothing, and one every program gets wrong is usually beyond
 * the reach of a prompt edit; the instructive ones are where one program
 * succeeded and another failed, because the pair is a controlled experiment
 * the reflection model can read directly.
 *
 * Ranked on the max-to-min gap first, then the best score, then the max-to-avg
 * gap: widest disagreement first, ties broken toward instances where something
 * actually worked, since a bucket whose best rollout is bad has no success to
 * generalize from.
 */
export function buildBuckets<Datum, Output>(args: {
  batch: readonly Datum[];
  samples: readonly SimbaSample<Output>[];
}): SimbaBucket<Datum, Output>[] {
  const { batch, samples } = args;

  const buckets = batch.map((datum, index) => {
    const rollouts = samples
      .map((sample) => ({
        programIndex: sample.programIndex,
        score: sample.scores[index] ?? 0,
        ...(sample.outputs === undefined
          ? {}
          : { output: sample.outputs[index] }),
        ...(sample.feedback === undefined
          ? {}
          : { feedback: sample.feedback[index] }),
      }))
      .sort((a, b) => b.score - a.score);

    const scores = rollouts.map((rollout) => rollout.score);
    const maxScore = Math.max(...scores);
    const minScore = Math.min(...scores);
    const average =
      scores.reduce((total, score) => total + score, 0) / scores.length;

    return {
      index,
      datum,
      rollouts,
      maxToMinGap: maxScore - minScore,
      maxScore,
      maxToAvgGap: maxScore - average,
    };
  });

  return buckets.sort(
    (a, b) =>
      b.maxToMinGap - a.maxToMinGap ||
      b.maxScore - a.maxScore ||
      b.maxToAvgGap - a.maxToAvgGap,
  );
}

/**
 * Linear-interpolated percentile, matching numpy's default so the thresholds
 * behave the way the reference implementation's do.
 */
export function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((a, b) => a - b);
  const position = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);

  if (lower === upper) {
    return sorted[lower] as number;
  }
  return (
    (sorted[lower] as number) +
    (position - lower) * ((sorted[upper] as number) - (sorted[lower] as number))
  );
}

/**
 * Softmax weights over program scores, for picking which program to mutate
 * next. Shifted by the maximum before exponentiating — the same distribution,
 * but a score scale the caller chose freely cannot overflow it.
 */
export function softmaxWeights(
  scores: readonly number[],
  temperature: number,
): number[] {
  if (scores.length === 0) {
    return [];
  }

  const highest = Math.max(...scores);
  return scores.map((score) => Math.exp((score - highest) / temperature));
}

/**
 * The `k` highest scoring programs, with the baseline forced into the pool.
 *
 * Keeping the baseline is what makes the search recoverable: every candidate
 * descends from a program already in the pool, so a pool that has drifted into
 * a bad region has nothing left to climb back from.
 */
export function topKPlusBaseline(args: {
  scores: readonly number[];
  k: number;
}): number[] {
  const { scores, k } = args;

  const ranked = scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((entry) => entry.index);

  if (ranked.length > 0 && !ranked.includes(0)) {
    ranked[ranked.length - 1] = 0;
  }
  return [...new Set(ranked)];
}

/**
 * A Poisson draw by Knuth's method, used to decide how many demonstrations to
 * drop before a mutation. Random rather than fixed so a candidate can shed a
 * demo that is no longer earning its place — nothing else in the loop ever
 * removes one, and a block that only grows eventually crowds out the
 * instruction it was meant to support.
 */
export function samplePoisson(rng: Rng, lambda: number): number {
  if (lambda <= 0) {
    return 0;
  }

  const limit = Math.exp(-lambda);
  let draws = 0;
  let product = 1;

  do {
    draws += 1;
    product *= rng.next();
  } while (product > limit);

  return draws - 1;
}

/**
 * Positions spread evenly across a sequence, first and last included.
 *
 * The step winners are held back and only a sample of them is scored on the
 * full validation set at the end, because scoring every one of them costs more
 * than the search did. Sampling evenly rather than taking the last few keeps
 * early winners in the running: minibatch scores are noisy, and the run's
 * genuine best is often not its most recent.
 */
export function evenlySpacedIndices(args: {
  length: number;
  count: number;
}): number[] {
  const { length, count } = args;

  if (length <= 1 || count <= 1) {
    return [0];
  }

  const last = length - 1;
  const indices = Array.from({ length: count }, (_, position) =>
    Math.round((position * last) / (count - 1)),
  );
  return [...new Set(indices)];
}
